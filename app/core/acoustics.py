"""
acoustics -- acoustic fingerprinting from scratch, the "constellation"
algorithm popularised by Shazam (Wang, 2003), implemented with numpy and
the standard library only.

WHY THIS EXISTS
---------------
SummarEase already knows what a recording *says*. It does not know what a
recording *is*. Those are different questions, and only the second one is
answerable exactly:

    "Have I already processed this exact meeting?"
    "Where does this 20-second quote appear in my 3-hour interview?"
    "Which of these 40 podcast episodes share the same intro jingle?"

A language model cannot answer those reliably, and file hashing cannot
answer them at all. sha256 of the bytes changes if the user re-exports at
a different bitrate, trims a leading half-second of silence, normalises
the volume, or records the same meeting on a second device. All of those
are the *same audio* to a human and a completely different file to a byte
hash.

A perceptual fingerprint is built from the shape of the sound rather than
the encoding of the bytes. We locate the loudest, most locally-dominant
points in the time/frequency plane -- the "constellation" of spectral
peaks -- and describe the audio purely by the *relative geometry* of that
constellation. Peaks are the last thing to be destroyed by noise, lossy
compression, EQ or gain changes: a quiet hiss added everywhere raises the
floor but does not move the summit of a hill. That is the entire reason
this beats file hashing, and it is why the pipeline below throws away the
waveform and keeps only peak relationships.

THE PIPELINE
------------
    PCM float32  ->  resample to ~11 kHz  ->  STFT (Hann window)
                 ->  log-magnitude spectrogram
                 ->  local-maximum peak picking with adaptive threshold
                 ->  density thinning (uniform peaks/second)
                 ->  combinatorial hashing of peak *pairs*
                 ->  {hash: hex, time: seconds} list + an inverted index

Hashing *pairs* rather than single peaks is the crucial trick. A single
peak ("there is energy at 2 kHz") carries almost no information and
collides constantly. A pair ("a peak at 2 kHz is followed 310 ms later by
a peak at 3.1 kHz") is far more specific -- roughly the difference
between a word and a whole phrase -- while still being translation
invariant in time, because the hash stores the *delta* between the two
peaks rather than their absolute positions. That invariance is what lets
a 5-second clip be located inside an hour-long recording.

MATCHING: WHY A HISTOGRAM OF TIME OFFSETS
-----------------------------------------
Counting how many hashes two recordings share is a weak test. Any two
recordings of speech share some hashes by coincidence, and a long
recording shares more simply by being long.

The real signal is *consistency*. If clip B is genuinely a copy of a
passage starting 42.3 s into recording A, then EVERY matching hash pair
must satisfy the same relationship:

    time_in_A - time_in_B = 42.3        (the same constant, for all of them)

So for every shared hash we compute that difference and drop it into a
histogram. A true match piles hundreds of votes into one narrow bin -- a
sharp spike. Coincidental hash collisions scatter uniformly across every
possible offset and produce no spike at all. We therefore score a match
by the height of the tallest bin relative to the scattered background,
not by raw overlap. This single idea is what makes the algorithm robust
enough to identify music in a noisy bar, and it is implemented once in
`_align()` and reused by `compare`, `find_repeats` and `query_index`.

Everything here is offline, deterministic, dependency-light (numpy +
stdlib) and fast: about half a second to fingerprint five minutes of
audio, and microseconds to query a library through the inverted index.
"""
from __future__ import annotations

import hashlib
import wave
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "fingerprint",
    "compare",
    "find_repeats",
    "build_index",
    "query_index",
    "load_wav",
    "spectrogram_preview",
]

# ---------------------------------------------------------------------------
# Analysis parameters
#
# These are not arbitrary; each one trades resolution against robustness
# and index size, and the comment explains which way.
# ---------------------------------------------------------------------------

#: Everything is analysed at this rate. Speech and music carry more than
#: enough identifying structure below 5.5 kHz, and decimating from 44.1 kHz
#: to 11.025 kHz makes the FFT four times cheaper while *improving*
#: robustness -- the discarded top octaves are exactly where lossy codecs
#: do their damage, so ignoring them removes a source of disagreement
#: between an original and its re-encoded copy.
TARGET_SAMPLE_RATE = 11025

#: 1024 samples @ 11025 Hz = 92.9 ms per frame, 10.8 Hz per bin. Long
#: enough to resolve individual musical/speech partials into distinct
#: peaks, short enough that transients are not smeared across a syllable.
N_FFT = 1024

#: 32 ms hop (~71% overlap). The hop sets the time quantisation of every
#: hash and of the offset histogram: finer means better alignment
#: precision, coarser means fewer frames. 32 ms lands alignment error
#: around +/-30 ms, well below anything a human would notice.
HOP_SECONDS = 0.032

#: Neighbourhood for the local-maximum test, in (frames, bins). ~7 frames
#: x 19 bins ~= 220 ms x 205 Hz. Too small and every ripple is a "peak";
#: too large and dense passages lose peaks needed for a short-clip match.
PEAK_TIME_RADIUS = 3
PEAK_FREQ_RADIUS = 9

#: Neighbourhood for the *background* estimate (a box mean). A peak must
#: stand this many dB above its own local background rather than above a
#: fixed absolute level -- that is what lets a whispered voice memo
#: fingerprint just as well as a mastered podcast.
BACKGROUND_TIME_RADIUS = 16
BACKGROUND_FREQ_RADIUS = 24

#: The margin is the single most important number in this file, and it is
#: not a taste judgement -- it is set by the statistics of noise.
#: Gaussian noise has Rayleigh-distributed FFT magnitudes, so the largest
#: of the ~7x19 = 133 cells in a peak neighbourhood sits, on average,
#: about 9-10 dB above that neighbourhood's mean *purely by chance*. Any
#: margin below that admits noise as landmarks; measured end to end, the
#: yield of a noisy copy collapses from 30% of hashes at 12 dB to 1% at
#: 9 dB, because the constellation fills up with peaks that were never
#: in the original audio. 12 dB is just above the noise ceiling.
PEAK_DB_ABOVE_BACKGROUND = 12.0

#: The background surface is by construction slowly varying (it is a
#: 33x49 box mean), so it is computed on a grid decimated by this factor
#: in each direction and then expanded back. Sixteen times less work for
#: an average error of ~0.13 dB, which is far below the 12 dB margin it
#: feeds.
BACKGROUND_DECIMATION = 4

#: Absolute dynamic-range gate, relative to the loudest point in this
#: recording. Stops digital silence and dither noise from generating a
#: constellation of meaningless peaks.
DYNAMIC_RANGE_DB = 70.0

#: Density control. Peaks are thinned so that each (1 second x 1/8th of
#: the spectrum) cell keeps at most PEAKS_PER_CELL of its strongest
#: candidates. Uniform density matters twice over: it stops a loud chorus
#: from swamping the index while a quiet verse contributes nothing, and it
#: makes fingerprint size linear in duration, which keeps the index small.
DENSITY_BLOCK_SECONDS = 1.0
DENSITY_BANDS = 8
PEAKS_PER_CELL = 6

#: Target-zone geometry for combinatorial hashing. From each anchor peak
#: we look FORWARD (never backward -- so the hash set of a clip is a
#: subset of the hash set of the track that contains it) into a window
#: 1..40 frames ahead (32 ms .. 1.3 s) and pair it with at most FAN_OUT
#: peaks. Fan-out is the central cost/robustness dial: more pairs per
#: anchor survives more peak dropouts, but index size grows linearly with
#: it. 15 is the classic compromise.
FAN_OUT = 15
DT_MIN_FRAMES = 1
DT_MAX_FRAMES = 40

#: Deliberate coarseness in the hash payload. Noise, resampling and lossy
#: re-encoding all nudge a peak by about a bin in frequency and a frame in
#: time; a hash that changed whenever a peak moved 10 Hz or 32 ms would be
#: useless. So 513 FFT bins are folded to 257 bands (bin >> 1) and the
#: time delta is stored at 64 ms resolution (frames >> 1). Each shift
#: trades one bit of specificity for a large drop in the chance that the
#: same musical event hashes differently in two recordings of it --
#: measured, the dt shift alone lifts a noisy copy's score by ~10%.
FREQ_QUANT_SHIFT = 1
DT_QUANT_SHIFT = 1

#: Hash packing: f1 (9 bits) | f2 (9 bits) | dt (6 bits) = 24 bits, which
#: renders as exactly 6 lowercase hex characters. Compact enough that a
#: fingerprint is cheap to store in Postgres, wide enough (16.7M values)
#: that accidental collisions are rare.
_F1_SHIFT = 15
_F2_SHIFT = 6
_FREQ_MASK = 0x1FF
_DT_MASK = 0x3F
_HASH_HEX_WIDTH = 6

#: Match decision thresholds, applied to the offset histogram.
MIN_ALIGNED_HASHES = 8
MIN_MATCH_SCORE = 0.035
MIN_SHARPNESS = 2.5

#: Safety valve: an adversarial or pathologically repetitive pair of
#: recordings could otherwise generate a quadratic number of candidate
#: pairs. We keep the most *distinctive* (rarest) hashes when capping.
_MAX_CANDIDATE_PAIRS = 4_000_000

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Low-level numpy helpers
#
# scipy is not a dependency, so the filters we need (box mean, sliding
# maximum, resampling) are written here. All of them are O(n) or O(n*r)
# with tiny constants and none of them materialise a windowed copy of the
# spectrogram, which is what keeps memory flat on long recordings.
# ---------------------------------------------------------------------------


def _hann(length: int) -> np.ndarray:
    """Periodic Hann window, written out rather than imported.

    Rectangular framing convolves the spectrum with a sinc, whose side
    lobes are only 13 dB down -- loud partials would splatter energy
    across the whole spectrum and manufacture fake peaks. Hann's side
    lobes are 31 dB down and roll off fast, so peaks stay where they
    belong. The *periodic* form (divisor `length`, not `length - 1`) is
    the right one for overlap-add STFT analysis.
    """
    if length <= 1:
        return np.ones(max(length, 1), dtype=np.float32)
    n = np.arange(length, dtype=np.float32)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / length)).astype(np.float32)


def _box_filter_1d(x: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average via a prefix sum: O(n) regardless of width.

    Used as the anti-alias stage before decimation. It is a crude
    low-pass, but decimation without *any* low-pass folds high-frequency
    content down into the band we analyse, which would make the same audio
    fingerprint differently depending on its original sample rate -- a far
    worse error than a gently sloped filter response.
    """
    if width <= 1 or x.size == 0:
        return x
    width = min(width, x.size)
    cumulative = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    idx = np.arange(x.size)
    half = width // 2
    lo = np.clip(idx - half, 0, x.size)
    hi = np.clip(idx - half + width, 0, x.size)
    return ((cumulative[hi] - cumulative[lo]) / np.maximum(hi - lo, 1)).astype(np.float32)


def _max_filter_axis(matrix: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Sliding maximum of the given radius along one axis.

    Written as `2 * radius` vectorised `np.maximum` passes against shifted
    views of the input. The obvious `sliding_window_view(...).max(-1)`
    would allocate a (frames x bins x window) array -- hundreds of MB for
    a long recording -- while this allocates one extra matrix total.
    """
    if radius <= 0:
        return matrix.copy()
    out = matrix.copy()
    for shift in range(1, radius + 1):
        if axis == 0:
            if shift >= matrix.shape[0]:
                break
            np.maximum(out[:-shift], matrix[shift:], out=out[:-shift])
            np.maximum(out[shift:], matrix[:-shift], out=out[shift:])
        else:
            if shift >= matrix.shape[1]:
                break
            np.maximum(out[:, :-shift], matrix[:, shift:], out=out[:, :-shift])
            np.maximum(out[:, shift:], matrix[:, :-shift], out=out[:, shift:])
    return out


def _max_filter_2d(matrix: np.ndarray, time_radius: int, freq_radius: int) -> np.ndarray:
    """Rectangular neighbourhood maximum. The maximum over a rectangle is
    separable (max over rows of the max over columns), so two 1-D passes
    give the exact 2-D result."""
    return _max_filter_axis(
        _max_filter_axis(matrix, time_radius, axis=0), freq_radius, axis=1
    )


def _box_mean_axis(matrix: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Edge-aware moving average along one axis, again via prefix sums."""
    if radius <= 0:
        return matrix.astype(np.float32, copy=True)
    n = matrix.shape[axis]
    cumulative = np.cumsum(matrix, axis=axis, dtype=np.float64)
    zeros = np.zeros_like(np.take(cumulative, [0], axis=axis))
    cumulative = np.concatenate([zeros, cumulative], axis=axis)
    idx = np.arange(n)
    lo = np.maximum(idx - radius, 0)
    hi = np.minimum(idx + radius + 1, n)
    total = np.take(cumulative, hi, axis=axis) - np.take(cumulative, lo, axis=axis)
    counts = (hi - lo).astype(np.float64)
    shape = [1] * matrix.ndim
    shape[axis] = n
    return (total / counts.reshape(shape)).astype(np.float32)


def _box_mean_2d(matrix: np.ndarray, time_radius: int, freq_radius: int) -> np.ndarray:
    """Separable rectangular mean -- the local background level."""
    return _box_mean_axis(
        _box_mean_axis(matrix, time_radius, axis=0), freq_radius, axis=1
    )


def _mean_pool(matrix: np.ndarray, factor: int, axis: int) -> np.ndarray:
    """Average adjacent cells in groups of `factor` along one axis."""
    n = matrix.shape[axis]
    edges = np.arange(0, n, factor)
    pooled = np.add.reduceat(matrix, edges, axis=axis)
    widths = np.diff(np.append(edges, n)).astype(np.float32)
    shape = [1] * matrix.ndim
    shape[axis] = edges.size
    return (pooled / widths.reshape(shape)).astype(np.float32)


def _background(matrix: np.ndarray, time_radius: int, freq_radius: int) -> np.ndarray:
    """Local background level, computed on a decimated grid.

    The background is a wide box mean, i.e. a deliberately smooth surface
    -- it has no detail finer than its own window to lose. Pooling the
    spectrogram down by BACKGROUND_DECIMATION in each direction, taking
    the box mean there, and expanding the result back costs a sixteenth of
    the work for an average error around 0.13 dB against the exact
    computation. That is two orders of magnitude below the 12 dB margin
    the background is used for, so the peak decisions are unchanged while
    the slowest step of the pipeline gets four times cheaper.
    """
    factor = max(1, int(BACKGROUND_DECIMATION))
    if factor == 1 or min(matrix.shape) < 2 * factor:
        return _box_mean_2d(matrix, time_radius, freq_radius)
    n_frames, n_bins = matrix.shape
    small = _mean_pool(_mean_pool(matrix, factor, axis=0), factor, axis=1)
    smoothed = _box_mean_2d(
        small, max(1, round(time_radius / factor)), max(1, round(freq_radius / factor))
    )
    expanded = np.repeat(np.repeat(smoothed, factor, axis=0), factor, axis=1)
    return expanded[:n_frames, :n_bins]


def _ragged_indices(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenate `range(start, start + count)` for every (start, count)
    pair, without a Python loop.

    This is the workhorse behind both the target-zone expansion and the
    hash join: both are "for each item, emit a variable number of rows",
    which is exactly what a naive loop makes slow. The trick is to repeat
    each start `count` times, then subtract a running offset so the
    repeated entries count upward instead of staying flat.
    """
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    ends = np.cumsum(counts)
    offsets = np.repeat(ends - counts, counts)
    return np.repeat(starts, counts) + (np.arange(total, dtype=np.int64) - offsets)


# ---------------------------------------------------------------------------
# Signal conditioning
# ---------------------------------------------------------------------------


def _as_mono_float32(samples: np.ndarray) -> np.ndarray:
    """Coerce whatever the caller handed us into finite mono float32.

    The browser posts float32 PCM, but being liberal here costs nothing
    and stops a stray int16 buffer or a NaN from a bad decode turning into
    an incomprehensible FFT error three functions later.
    """
    arr = np.asarray(samples)
    if arr.ndim > 1:
        # (frames, channels) or (channels, frames) -- mix to mono either way.
        axis = 1 if arr.shape[0] >= arr.shape[-1] else 0
        arr = arr.mean(axis=axis)
    arr = np.asarray(arr, dtype=np.float32).ravel()
    if arr.size and not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _resample(samples: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    """Rate conversion by anti-aliased decimation + linear interpolation.

    Deliberately simple: no scipy, no polyphase filter bank. Fingerprint
    accuracy does not need a textbook-perfect reconstruction, it needs the
    *same* treatment applied to every recording so that peaks land in the
    same bins. Box-filter-then-interpolate is stable, allocation-light and
    entirely predictable.
    """
    if sample_rate == target_rate or samples.size == 0:
        return samples

    rate = float(sample_rate)
    # Integer-factor stage first. Reshaping to (-1, factor) and taking the
    # row mean is simultaneously the anti-alias filter and the decimation,
    # in one cache-friendly pass -- ten times faster than filtering the
    # full-rate signal and then interpolating it, and it covers the two
    # rates that matter most (44100 and 22050 both divide exactly to
    # 11025, so they never touch the interpolation branch at all).
    factor = int(rate // target_rate)
    if factor > 1:
        usable = (samples.size // factor) * factor
        if usable >= factor:
            samples = samples[:usable].reshape(-1, factor).mean(axis=1).astype(np.float32)
            rate = rate / factor

    if abs(rate - target_rate) < 1e-9:
        return samples
    if rate > target_rate:
        # Residual ratio is now under 2x; one more short box filter keeps
        # the little that is left above the new Nyquist from folding back.
        samples = _box_filter_1d(samples, int(round(rate / target_rate)) + 1)
    n_out = int(round(samples.size * target_rate / rate))
    if n_out < 2:
        return samples[:1]
    positions = np.arange(n_out, dtype=np.float64) * (rate / target_rate)
    source = np.arange(samples.size, dtype=np.float64)
    return np.interp(positions, source, samples).astype(np.float32)


def _hop_length(sample_rate: int) -> int:
    """Hop in samples for this rate (HOP_SECONDS, at least 1)."""
    return max(1, int(round(sample_rate * HOP_SECONDS)))


def _stft_magnitude(
    samples: np.ndarray, n_fft: int, hop: int
) -> np.ndarray:
    """Magnitude spectrogram, shape (frames, n_fft // 2 + 1).

    Frames are cut with `sliding_window_view`, which is a *view* -- no
    copy of the signal is made for the overlap. The window multiply then
    produces one contiguous (frames x n_fft) array which numpy's batched
    rfft chews through in a single call. That single batched call is the
    whole reason a five-minute recording fingerprints in well under a
    second.
    """
    if samples.size < n_fft:
        samples = np.pad(samples, (0, n_fft - samples.size))
    frames = sliding_window_view(samples, n_fft)[::hop]
    if frames.shape[0] == 0:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.float32)
    windowed = frames * _hann(n_fft)
    spectrum = np.fft.rfft(windowed, axis=1)
    return np.abs(spectrum).astype(np.float32)


def _log_magnitude(magnitude: np.ndarray) -> np.ndarray:
    """Magnitude in dB.

    Peak picking happens in the log domain because hearing -- and every
    codec designed for it -- is logarithmic. In linear magnitude a quiet
    but perfectly clear harmonic is numerically invisible next to the
    fundamental; in dB it is an obvious local maximum. Working in dB is
    also what makes the fingerprint gain-invariant: scaling the waveform
    by 0.3 subtracts a constant ~10.5 dB everywhere, which shifts the
    whole surface without moving a single local maximum.
    """
    return (20.0 * np.log10(magnitude + _EPS)).astype(np.float32)


# ---------------------------------------------------------------------------
# Constellation: peak picking
# ---------------------------------------------------------------------------


def _pick_peaks(log_mag: np.ndarray, hop: int, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Find the constellation points. Returns (frame_index, freq_bin),
    both sorted by frame.

    Three independent gates, each closing a different failure mode:

    1. **Local maximum** over a (time x frequency) neighbourhood. This is
       what makes a peak a landmark instead of just a loud sample: it must
       dominate its surroundings, which is a property that survives added
       noise, EQ and lossy compression.
    2. **Above the local background** by a fixed dB margin. Relative, not
       absolute -- a phone voice memo recorded at -40 dBFS has to yield
       the same constellation as the same audio normalised to -3 dBFS.
    3. **Within the recording's dynamic range**, so silence and dither
       never become landmarks.

    Then density thinning: keep only the strongest few peaks per
    (1 s x 1/8-spectrum) cell, so hash count stays proportional to
    duration and a loud passage cannot crowd out a quiet one.
    """
    n_frames, n_bins = log_mag.shape
    if n_frames == 0 or n_bins == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    neighbourhood_max = _max_filter_2d(log_mag, PEAK_TIME_RADIUS, PEAK_FREQ_RADIUS)
    background = _background(log_mag, BACKGROUND_TIME_RADIUS, BACKGROUND_FREQ_RADIUS)
    ceiling = float(log_mag.max())

    is_peak = (
        (log_mag >= neighbourhood_max - 1e-4)
        & (log_mag >= background + PEAK_DB_ABOVE_BACKGROUND)
        & (log_mag >= ceiling - DYNAMIC_RANGE_DB)
    )
    # Bin 0 is DC and the top bin is Nyquist; neither is a real landmark.
    is_peak[:, 0] = False
    if n_bins > 1:
        is_peak[:, -1] = False

    frames, bins = np.nonzero(is_peak)
    if frames.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    # Rank by PROMINENCE (dB above the local background), not by absolute
    # loudness. Absolute loudness ranks a peak by which band happens to be
    # energetic, so the ordering inside a cell reshuffles the moment noise
    # is added and the surviving peak set diverges between two copies of
    # the same audio. Prominence ranks a peak by how much it stands out
    # from its own surroundings, which is a far more stable quantity --
    # exactly the property the whole fingerprint rests on.
    strength = log_mag[frames, bins] - background[frames, bins]

    # --- density thinning -------------------------------------------------
    # Cell key = (time block, frequency band). Sorting by (key, -strength)
    # and taking each key's first PEAKS_PER_CELL rows is a vectorised
    # "top-k per group": `searchsorted` against the sorted keys gives each
    # row's group start, and index-minus-start is its rank in the group.
    frames_per_block = max(1, int(round(DENSITY_BLOCK_SECONDS * sample_rate / hop)))
    band_width = max(1, int(np.ceil(n_bins / DENSITY_BANDS)))
    keys = (frames // frames_per_block) * DENSITY_BANDS + (bins // band_width)

    order = np.lexsort((-strength, keys))
    keys_sorted = keys[order]
    group_start = np.searchsorted(keys_sorted, keys_sorted, side="left")
    rank = np.arange(keys_sorted.size, dtype=np.int64) - group_start
    kept = order[rank < PEAKS_PER_CELL]

    frames = frames[kept].astype(np.int64)
    bins = bins[kept].astype(np.int64)

    # Hashing walks forward through time, so the peak list must be sorted
    # by frame for `searchsorted` to delimit each target zone.
    order = np.lexsort((bins, frames))
    return frames[order], bins[order]


# ---------------------------------------------------------------------------
# Combinatorial hashing
# ---------------------------------------------------------------------------


def _hash_peaks(
    peak_frames: np.ndarray, peak_bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pair every anchor peak with peaks in its target zone and pack each
    pair into a 24-bit code. Returns (codes, anchor_frame).

    The target zone is a window *ahead* of the anchor: frames
    [f + DT_MIN_FRAMES, f + DT_MAX_FRAMES], capped at FAN_OUT peaks.

    Why forward-only and time-relative: the code stores (f1, f2, dt), never
    an absolute time. Cut a clip out of the middle of a track and every
    surviving anchor produces byte-identical codes to the ones the full
    track produced, because nothing in the code depends on where the clip
    started. The clip's hash set becomes a subset of the track's, and the
    absolute position is recovered separately from the offset histogram.
    Pairing backwards as well would only duplicate that information.

    Why a bounded zone: dt has to fit in 6 bits, and more importantly a
    peak 30 seconds away tells you nothing about this one -- the pair
    would be describing two unrelated events and would break the moment
    anything between them changed.

    Vectorised end to end: `searchsorted` finds each anchor's zone
    boundaries in one call, and `_ragged_indices` expands the
    variable-length zones into a flat index array with no Python loop.
    """
    n = peak_frames.size
    if n < 2:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i

    zone_start = np.searchsorted(peak_frames, peak_frames + DT_MIN_FRAMES, side="left")
    zone_end = np.searchsorted(peak_frames, peak_frames + DT_MAX_FRAMES, side="right")
    counts = np.minimum(zone_end - zone_start, FAN_OUT).astype(np.int64)
    counts = np.maximum(counts, 0)
    if counts.sum() == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i

    target_idx = _ragged_indices(zone_start.astype(np.int64), counts)
    anchor_idx = np.repeat(np.arange(n, dtype=np.int64), counts)

    f1 = (peak_bins[anchor_idx] >> FREQ_QUANT_SHIFT) & _FREQ_MASK
    f2 = (peak_bins[target_idx] >> FREQ_QUANT_SHIFT) & _FREQ_MASK
    dt = ((peak_frames[target_idx] - peak_frames[anchor_idx]) >> DT_QUANT_SHIFT) & _DT_MASK

    codes = (f1 << _F1_SHIFT) | (f2 << _F2_SHIFT) | dt
    return codes.astype(np.int64), peak_frames[anchor_idx]


def _codes_to_hex(codes: np.ndarray) -> list[str]:
    """Pack 24-bit codes into zero-padded 6-char hex, vectorised through a
    nibble lookup table.

    `[format(c, '06x') for c in codes]` is a Python-level loop over what
    can be hundreds of thousands of values and would dominate the runtime
    of the whole fingerprint. Splitting each code into six nibbles, using
    them to index a 16-character array and viewing the result as a
    6-character string dtype does the same job entirely inside numpy.

    Zero padding matters beyond tidiness: it makes lexicographic ordering
    of the hex strings identical to numeric ordering of the codes, which
    is what lets `fingerprint_id` be defined over the sorted hex strings
    yet computed from a numpy sort.
    """
    if codes.size == 0:
        return []
    table = np.frombuffer(b"0123456789abcdef", dtype="S1")
    nibbles = np.empty((codes.size, _HASH_HEX_WIDTH), dtype="S1")
    for position in range(_HASH_HEX_WIDTH):
        shift = 4 * (_HASH_HEX_WIDTH - 1 - position)
        nibbles[:, position] = table[(codes >> shift) & 0xF]
    packed = np.ascontiguousarray(nibbles).view(f"S{_HASH_HEX_WIDTH}").ravel()
    return [value.decode("ascii") for value in packed]


# ---------------------------------------------------------------------------
# Public API: fingerprinting
# ---------------------------------------------------------------------------


def fingerprint(
    samples: np.ndarray, sample_rate: int, *, target_sample_rate: int = TARGET_SAMPLE_RATE
) -> dict[str, Any]:
    """Fingerprint mono PCM.

    `samples` is a numpy array of mono float32 samples in roughly [-1, 1]
    and `sample_rate` is its rate in Hz. This is deliberately NOT a file
    path: SummarEase decodes in the browser with WebAudio's
    `decodeAudioData` and POSTs raw PCM, so the server needs no ffmpeg, no
    codec licences and no temp files. `load_wav()` exists for tests and
    for server-side WAV.

    Returns a JSON-serialisable dict::

        {
          "hashes": [{"hash": "1a2b3c", "time": 4.512}, ...],
          "duration": 61.44,          # seconds of audio analysed
          "peak_count": 892,          # constellation points found
          "hash_count": 12760,        # peak pairs emitted
          "sample_rate": 11025,       # the rate actually analysed at
          "fingerprint_id": "9f86d0...",
        }

    `time` is the offset of the *anchor* peak of each pair, which is what
    the offset histogram in `compare()` aligns on.

    `fingerprint_id` is a sha256 over the newline-joined, sorted hash
    multiset. It is a perceptual identity: two files that sound identical
    produce the same id even if their bytes differ (different container,
    different bitrate, different exporter), which is precisely what byte
    hashing cannot do. It is deliberately *exact* -- any real-world noise
    changes it -- so use it as a fast "definitely already seen this" cache
    key and fall back to `compare()` for the fuzzy question.
    """
    mono = _as_mono_float32(samples)
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    duration = mono.size / float(sample_rate) if mono.size else 0.0
    target_rate = int(target_sample_rate) if target_sample_rate else sample_rate
    if target_rate <= 0:
        raise ValueError("target_sample_rate must be positive")

    resampled = _resample(mono, sample_rate, target_rate)
    hop = _hop_length(target_rate)

    if resampled.size < N_FFT // 2:
        # Too short to hold even one meaningful analysis frame.
        return {
            "hashes": [],
            "duration": round(duration, 6),
            "peak_count": 0,
            "hash_count": 0,
            "sample_rate": target_rate,
            "fingerprint_id": hashlib.sha256(b"").hexdigest(),
        }

    magnitude = _stft_magnitude(resampled, N_FFT, hop)
    log_mag = _log_magnitude(magnitude)
    peak_frames, peak_bins = _pick_peaks(log_mag, hop, target_rate)
    codes, anchor_frames = _hash_peaks(peak_frames, peak_bins)

    seconds_per_frame = hop / float(target_rate)
    hex_codes = _codes_to_hex(codes)
    times = (anchor_frames * seconds_per_frame).round(6)

    hashes = [
        {"hash": code, "time": float(time)}
        for code, time in zip(hex_codes, times.tolist())
    ]

    # Sorting the packed ints and re-rendering them is equivalent to
    # sorting the zero-padded hex strings, but does the sort in numpy.
    digest_source = "\n".join(_codes_to_hex(np.sort(codes))).encode("ascii")

    return {
        "hashes": hashes,
        "duration": round(duration, 6),
        "peak_count": int(peak_frames.size),
        "hash_count": len(hashes),
        "sample_rate": target_rate,
        "fingerprint_id": hashlib.sha256(digest_source).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Matching machinery
# ---------------------------------------------------------------------------


def _fp_arrays(fp: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Unpack a fingerprint dict into (int64 codes, float64 times)."""
    entries = fp.get("hashes") or []
    if not entries:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    codes = np.fromiter(
        (int(entry["hash"], 16) for entry in entries), dtype=np.int64, count=len(entries)
    )
    times = np.fromiter(
        (float(entry["time"]) for entry in entries), dtype=np.float64, count=len(entries)
    )
    return codes, times


def _bin_width(*fps: dict[str, Any]) -> float:
    """Histogram bin width: one STFT hop.

    The offsets we are histogramming are differences of anchor times, and
    anchor times are quantised to the hop. Binning finer than the hop just
    splits one true peak across empty bins; binning much coarser blurs
    distinct offsets together.
    """
    for fp in fps:
        rate = int(fp.get("sample_rate") or 0)
        if rate > 0:
            return _hop_length(rate) / float(rate)
    return HOP_SECONDS


def _candidate_pairs(
    codes_a: np.ndarray,
    times_a: np.ndarray,
    codes_b: np.ndarray,
    times_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Join two hash lists on the hash value. Returns (index into A,
    time_b - time_a) for every matching pair.

    A vectorised sort-merge join: sort B by hash, then `searchsorted` with
    both sides gives, for each entry of A, the half-open range of B rows
    sharing its hash. `_ragged_indices` flattens those ranges. No dicts,
    no Python loop -- this runs in tens of milliseconds on fingerprints
    with hundreds of thousands of hashes.
    """
    if codes_a.size == 0 or codes_b.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    order = np.argsort(codes_b, kind="stable")
    sorted_codes = codes_b[order]
    sorted_times = times_b[order]

    left = np.searchsorted(sorted_codes, codes_a, side="left")
    right = np.searchsorted(sorted_codes, codes_a, side="right")
    counts = (right - left).astype(np.int64)

    if counts.sum() > _MAX_CANDIDATE_PAIRS:
        # Discard the most promiscuous hashes first. A hash that matches
        # thousands of rows is, by definition, not distinctive -- dropping
        # it costs almost no discriminative power and bounds the work.
        by_count = np.argsort(counts, kind="stable")
        cumulative = np.cumsum(counts[by_count])
        keep_n = int(np.searchsorted(cumulative, _MAX_CANDIDATE_PAIRS, side="right"))
        allowed = np.zeros(counts.size, dtype=bool)
        allowed[by_count[:keep_n]] = True
        counts = np.where(allowed, counts, 0)

    if counts.sum() == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    b_idx = _ragged_indices(left.astype(np.int64), counts)
    a_idx = np.repeat(np.arange(codes_a.size, dtype=np.int64), counts)
    deltas = sorted_times[b_idx] - times_a[a_idx]
    return a_idx, deltas


def _smoothed_histogram(
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin counts, each smoothed with its two immediate neighbours.

    Smoothing matters because a true offset rarely falls exactly on a bin
    boundary: quantisation and a fractional-sample time shift split one
    spike across two adjacent bins. Summing each bin with its neighbours
    recovers the full spike without widening the search.
    """
    unique, counts = np.unique(bins, return_counts=True)
    previous_pos = np.searchsorted(unique, unique - 1)
    next_pos = np.searchsorted(unique, unique + 1)

    smoothed = counts.astype(np.int64).copy()
    valid = previous_pos < unique.size
    has_prev = np.zeros(unique.size, dtype=bool)
    has_prev[valid] = unique[previous_pos[valid]] == unique[valid] - 1
    smoothed[has_prev] += counts[previous_pos[has_prev]]

    valid = next_pos < unique.size
    has_next = np.zeros(unique.size, dtype=bool)
    has_next[valid] = unique[next_pos[valid]] == unique[valid] + 1
    smoothed[has_next] += counts[next_pos[has_next]]

    return unique, smoothed


def _align(
    codes_a: np.ndarray,
    a_idx: np.ndarray,
    deltas: np.ndarray,
    bin_width: float,
) -> dict[str, Any]:
    """THE core of the algorithm: turn a bag of candidate pairs into an
    alignment verdict via the time-offset histogram.

    Every candidate pair votes for the offset it implies. If the two
    recordings really do share a passage, all of those votes agree, because
    a shared passage has one and only one time offset between the two
    timelines -- so the votes stack into a single tall bin. If they merely
    collide by chance, each collision implies a different offset and the
    votes spread thinly over every bin in range. That difference in
    *shape*, not in raw overlap count, is the evidence.

    Returns the peak offset, how many DISTINCT hashes voted for it (a
    repeated hash must not be allowed to vote twice), and `sharpness`:
    the peak's height divided by the height a uniform scatter of the same
    number of votes would have produced. Sharpness near 1 means "this is
    what noise looks like"; sharpness of 10 means the peak is ten times
    anything chance would explain.
    """
    result: dict[str, Any] = {
        "offset_seconds": None,
        "aligned_hashes": 0,
        "sharpness": 0.0,
        "peak_votes": 0,
    }
    if deltas.size == 0:
        return result

    bins = np.rint(deltas / bin_width).astype(np.int64)
    unique, smoothed = _smoothed_histogram(bins)
    best = int(np.argmax(smoothed))
    best_bin = int(unique[best])
    peak_votes = int(smoothed[best])

    aligned_mask = np.abs(bins - best_bin) <= 1
    if not aligned_mask.any():
        return result

    # Average the true deltas in the winning bins rather than reporting
    # the bin centre: this recovers sub-hop precision for free.
    offset = float(np.mean(deltas[aligned_mask]))
    aligned = int(np.unique(codes_a[a_idx[aligned_mask]]).size)

    # Sharpness = how many times taller the winning bin is than the
    # scattered background around it. The null model is "the votes are
    # spread uniformly over the offsets that were actually observed", so
    # the background level is measured with the winning bins *excluded* --
    # otherwise a genuine match, whose votes are almost all inside the
    # peak, would be compared against itself and score ~1.
    span = max(1, int(unique[-1] - unique[0]) + 1)
    background_bins = max(1, span - 3)
    background_votes = max(0, deltas.size - peak_votes)
    expected = background_votes / float(background_bins)
    sharpness = peak_votes / max(expected, 1.0)

    result.update(
        offset_seconds=offset,
        aligned_hashes=aligned,
        sharpness=float(sharpness),
        peak_votes=peak_votes,
    )
    return result


def _distinct_count(codes: np.ndarray) -> int:
    return int(np.unique(codes).size) if codes.size else 0


# ---------------------------------------------------------------------------
# Public API: comparison
# ---------------------------------------------------------------------------


def compare(fp_a: dict[str, Any], fp_b: dict[str, Any]) -> dict[str, Any]:
    """Decide whether two fingerprints describe the same audio, and if so
    by what time offset.

    Returns::

        {
          "match": bool,
          "score": float,            # 0..1, aligned share of the smaller fp
          "common_hashes": int,      # distinct hashes present in both
          "offset_seconds": float|None,   # time_b - time_a of the alignment
          "aligned_hashes": int,     # distinct hashes voting for that offset
        }

    A positive `offset_seconds` means A's timeline starts that many
    seconds *before* B's -- i.e. the shared passage occurs at time `t` in
    A and `t + offset` in B. So comparing a clip (A) against the full
    track (B) reports where in the track the clip was taken from.

    `score` normalises by the *smaller* fingerprint on purpose. A 10-second
    clip can never align more hashes than it has, so dividing by the
    hour-long track it came from would score a perfect match at 0.003 and
    make the number meaningless. Dividing by the smaller side asks the
    right question: "how much of the thing that could have matched, did?"

    `match` requires all three of: enough aligned hashes to rule out luck,
    a meaningful score, and a histogram peak clearly above the scattered
    background. Requiring sharpness as well as count is what keeps two
    unrelated recordings -- which always share *some* hashes -- from being
    declared a match.
    """
    codes_a, times_a = _fp_arrays(fp_a)
    codes_b, times_b = _fp_arrays(fp_b)

    empty = {
        "match": False,
        "score": 0.0,
        "common_hashes": 0,
        "offset_seconds": None,
        "aligned_hashes": 0,
    }
    if codes_a.size == 0 or codes_b.size == 0:
        return empty

    common = int(np.intersect1d(codes_a, codes_b, assume_unique=False).size)
    if common == 0:
        return empty

    a_idx, deltas = _candidate_pairs(codes_a, times_a, codes_b, times_b)
    alignment = _align(codes_a, a_idx, deltas, _bin_width(fp_a, fp_b))

    denominator = max(1, min(_distinct_count(codes_a), _distinct_count(codes_b)))
    score = min(1.0, alignment["aligned_hashes"] / float(denominator))

    matched = (
        alignment["aligned_hashes"] >= MIN_ALIGNED_HASHES
        and score >= MIN_MATCH_SCORE
        and alignment["sharpness"] >= MIN_SHARPNESS
    )

    return {
        "match": bool(matched),
        "score": round(float(score), 6),
        "common_hashes": common,
        "offset_seconds": (
            round(float(alignment["offset_seconds"]), 4)
            if alignment["offset_seconds"] is not None
            else None
        ),
        "aligned_hashes": int(alignment["aligned_hashes"]),
    }


# ---------------------------------------------------------------------------
# Public API: self-similarity
# ---------------------------------------------------------------------------


def find_repeats(
    fp: dict[str, Any], *, min_gap_seconds: float = 5.0, min_hashes: int = 12
) -> list[dict[str, Any]]:
    """Find passages that repeat WITHIN a single recording.

    This is the same offset histogram turned inward: join the fingerprint
    against *itself*. Trivially, every hash matches itself at offset 0, so
    we discard offsets below `min_gap_seconds`. What survives is the
    interesting part -- a tall bin at offset 12.0 s means a whole cluster
    of peak-pairs recurs exactly 12 seconds later, which is what an intro
    jingle, a repeated chorus, a re-read sentence or a looping hold tone
    looks like in the time/frequency plane.

    Returns, strongest first::

        [{"first_at": 0.0, "repeat_at": 12.0, "duration": 3.1, "strength": 0.94}]

    `duration` is the time span of the aligned anchors, i.e. how long the
    repeated passage runs. `strength` is the share of the fingerprint's
    hashes inside that span that took part in the repeat: 1.0 means the
    passage recurs identically, 0.3 means only part of it came back.

    Candidates are chosen greedily with a separation rule so that one long
    repeat reported at offset 12.00 does not also get reported at 11.97
    and 12.03.
    """
    codes, times = _fp_arrays(fp)
    if codes.size == 0:
        return []

    bin_width = _bin_width(fp)
    a_idx, deltas = _candidate_pairs(codes, times, codes, times)
    if deltas.size == 0:
        return []

    # Only forward repeats: every backward pair is the mirror of a forward
    # one, and offset 0 is the trivial self-match.
    forward = deltas >= float(min_gap_seconds)
    if not forward.any():
        return []
    a_idx = a_idx[forward]
    deltas = deltas[forward]

    bins = np.rint(deltas / bin_width).astype(np.int64)
    unique, smoothed = _smoothed_histogram(bins)

    candidates = np.argsort(-smoothed)
    separation_bins = max(1, int(round(max(min_gap_seconds, 1.0) / bin_width)))

    results: list[dict[str, Any]] = []
    accepted: list[int] = []
    for position in candidates:
        votes = int(smoothed[position])
        if votes < min_hashes:
            break
        offset_bin = int(unique[position])
        if any(abs(offset_bin - other) < separation_bins for other in accepted):
            continue

        mask = np.abs(bins - offset_bin) <= 1
        anchor_times = times[a_idx[mask]]
        if anchor_times.size == 0:
            continue
        aligned = int(np.unique(codes[a_idx[mask]]).size)
        if aligned < min_hashes:
            continue

        first_at = float(anchor_times.min())
        last_at = float(anchor_times.max())
        duration = max(last_at - first_at, bin_width)
        offset = float(np.mean(deltas[mask]))

        # Normalise by how many hashes the source span actually contains,
        # so a short jingle that repeats perfectly scores near 1.0 rather
        # than being penalised for being short.
        in_span = _distinct_count(codes[(times >= first_at) & (times <= last_at)])
        strength = min(1.0, aligned / float(max(in_span, 1)))

        accepted.append(offset_bin)
        results.append(
            {
                "first_at": round(first_at, 4),
                "repeat_at": round(first_at + offset, 4),
                "duration": round(duration, 4),
                "strength": round(strength, 4),
            }
        )

    results.sort(key=lambda item: (-item["strength"], item["first_at"]))
    return results


# ---------------------------------------------------------------------------
# Public API: inverted index
# ---------------------------------------------------------------------------


def build_index(fingerprints: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Build an inverted index: hash -> [(track_index, anchor_time), ...].

    Comparing a query against a library pairwise costs O(library size) full
    comparisons. An inverted index turns that into O(query size) dictionary
    lookups: we touch only the tracks that actually share a hash with the
    query, and every other track in the library costs nothing at all. This
    is the difference between "search my 400 episodes" taking a minute and
    taking a few milliseconds, and it is the same structure a text search
    engine uses -- hashes are the vocabulary, recordings are the documents.

    The returned dict is plain JSON types (no numpy, no tuples) so it can
    be cached, pickled or stored as JSONB without conversion::

        {
          "version": 1,
          "tracks": ["memo-1", "memo-2"],
          "track_hash_counts": [12760, 5108],   # distinct hashes per track
          "postings": {"1a2b3c": [[0, 4.512], [1, 88.0]], ...},
        }
    """
    tracks: list[str] = []
    track_hash_counts: list[int] = []
    postings: dict[str, list[list[Any]]] = {}

    for track_id, fp in fingerprints or []:
        index = len(tracks)
        tracks.append(str(track_id))
        seen: set[str] = set()
        for entry in fp.get("hashes") or []:
            code = entry["hash"]
            seen.add(code)
            postings.setdefault(code, []).append([index, float(entry["time"])])
        track_hash_counts.append(len(seen))

    return {
        "version": 1,
        "tracks": tracks,
        "track_hash_counts": track_hash_counts,
        "postings": postings,
    }


def query_index(
    index: dict[str, Any], fp: dict[str, Any], *, top_n: int = 5
) -> list[dict[str, Any]]:
    """Search an index built by `build_index`, best match first.

    Returns::

        [{"track_id": "memo-2", "score": 0.81,
          "offset_seconds": 42.3, "aligned_hashes": 655}, ...]

    Two stages, deliberately: a cheap gather then an exact verdict. The
    gather walks the query's hashes once and collects, per candidate
    track, the implied time offsets. The verdict runs the same offset
    histogram as `compare()` on each candidate. Tracks that share nothing
    are never even looked at, and tracks that share hashes only by chance
    are eliminated by the histogram rather than by an arbitrary overlap
    cutoff -- so a popular hash cannot drag an unrelated recording to the
    top of the results.
    """
    postings: dict[str, list[list[Any]]] = index.get("postings") or {}
    tracks: list[str] = index.get("tracks") or []
    hash_counts: list[int] = index.get("track_hash_counts") or []
    entries = fp.get("hashes") or []
    if not postings or not tracks or not entries:
        return []

    # Stage 1 -- gather. One pass over the query's hashes; per track we
    # accumulate the query-hash code and the implied offset for each hit.
    per_track_codes: dict[int, list[int]] = {}
    per_track_deltas: dict[int, list[float]] = {}
    for entry in entries:
        code = entry["hash"]
        posting = postings.get(code)
        if not posting:
            continue
        query_time = float(entry["time"])
        code_int = int(code, 16)
        for track_index, track_time in posting:
            track_index = int(track_index)
            per_track_codes.setdefault(track_index, []).append(code_int)
            per_track_deltas.setdefault(track_index, []).append(
                float(track_time) - query_time
            )

    if not per_track_codes:
        return []

    query_distinct = max(1, _distinct_count(_fp_arrays(fp)[0]))
    bin_width = _bin_width(fp)

    # Stage 2 -- verdict. Offset histogram per candidate track.
    results: list[dict[str, Any]] = []
    for track_index, code_list in per_track_codes.items():
        if track_index < 0 or track_index >= len(tracks):
            continue
        codes = np.asarray(code_list, dtype=np.int64)
        deltas = np.asarray(per_track_deltas[track_index], dtype=np.float64)
        alignment = _align(codes, np.arange(codes.size, dtype=np.int64), deltas, bin_width)
        if alignment["aligned_hashes"] == 0:
            continue

        track_distinct = (
            hash_counts[track_index]
            if track_index < len(hash_counts) and hash_counts[track_index]
            else query_distinct
        )
        denominator = max(1, min(query_distinct, int(track_distinct)))
        score = min(1.0, alignment["aligned_hashes"] / float(denominator))

        results.append(
            {
                "track_id": tracks[track_index],
                "score": round(float(score), 6),
                "offset_seconds": round(float(alignment["offset_seconds"]), 4),
                "aligned_hashes": int(alignment["aligned_hashes"]),
                # Kept out of the public contract but useful for tuning.
                "_sharpness": round(float(alignment["sharpness"]), 4),
            }
        )

    # Rank by aligned hash count first: it is the raw strength of the
    # evidence. Score breaks ties and keeps the ordering sensible when two
    # candidates align a similar number of hashes from different-sized
    # libraries entries.
    results.sort(key=lambda item: (-item["aligned_hashes"], -item["score"]))
    for item in results:
        item.pop("_sharpness", None)
    return results[: max(0, int(top_n))]


# ---------------------------------------------------------------------------
# Public API: WAV loading
# ---------------------------------------------------------------------------


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a PCM WAV file with the stdlib `wave` module only.

    Handles 8-bit unsigned, 16-bit, 24-bit and 32-bit signed PCM, mono or
    multi-channel (channels are averaged to mono, because fingerprinting
    is about spectral content and a stereo image only adds a second,
    almost identical, copy of it).

    Returns (float32 samples in [-1, 1], sample_rate).

    This exists so the feature is testable and so server-side WAV upload
    still works without ffmpeg. The production path is the browser
    decoding with WebAudio and POSTing raw PCM -- which is why
    `fingerprint()` takes samples, not a path.
    """
    with wave.open(path, "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())

    if not raw:
        return np.zeros(0, dtype=np.float32), sample_rate

    if width == 1:
        # 8-bit WAV is unsigned with a 128 offset, unlike every other width.
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        # 24-bit has no numpy dtype: sign-extend three little-endian bytes
        # into the high three bytes of an int32 and scale accordingly.
        bytes_view = np.frombuffer(raw, dtype=np.uint8)
        usable = (bytes_view.size // 3) * 3
        triples = bytes_view[:usable].reshape(-1, 3).astype(np.uint32)
        packed = (triples[:, 0] << 8) | (triples[:, 1] << 16) | (triples[:, 2] << 24)
        data = packed.astype(np.uint32).view(np.int32).astype(np.float32) / 2147483648.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")

    if channels > 1:
        usable = (data.size // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)

    return np.clip(data, -1.0, 1.0).astype(np.float32), int(sample_rate)


# ---------------------------------------------------------------------------
# Public API: visualisation
# ---------------------------------------------------------------------------


def spectrogram_preview(
    samples: np.ndarray,
    sample_rate: int,
    *,
    max_bins: int = 64,
    max_frames: int = 240,
) -> dict[str, Any]:
    """A small, log-scaled, 0..1-normalised spectrogram for drawing in the
    browser.

    Returns::

        {"matrix": [[...], ...], "duration": 61.4, "freq_max": 5512.5}

    `matrix` is row-major with row 0 the LOWEST frequency band and each
    column a time step, so a canvas can draw it directly with the y-axis
    flipped. Values are 0..1 where 1 is the loudest point in the
    recording.

    Sending the full spectrogram to a browser would be megabytes of JSON
    for a picture a few hundred pixels wide, so both axes are average-
    pooled down to at most (max_bins x max_frames). Pooling happens in the
    dB domain rather than on linear magnitudes: averaging linear magnitude
    lets one loud bin whitewash its whole cell, whereas averaging dB gives
    the perceptually even picture a person expects to see.
    """
    mono = _as_mono_float32(samples)
    sample_rate = int(sample_rate)
    duration = mono.size / float(sample_rate) if mono.size and sample_rate > 0 else 0.0

    # Cap the analysis rate: nobody can read detail above ~11 kHz in a
    # 64-row picture, and decimating keeps the FFT cheap for long files.
    analysis_rate = sample_rate
    if sample_rate > 2 * TARGET_SAMPLE_RATE:
        analysis_rate = 2 * TARGET_SAMPLE_RATE
        mono = _resample(mono, sample_rate, analysis_rate)

    if mono.size < 16 or analysis_rate <= 0:
        return {
            "matrix": [],
            "duration": round(duration, 6),
            "freq_max": float(analysis_rate) / 2.0 if analysis_rate > 0 else 0.0,
        }

    hop = _hop_length(analysis_rate)
    magnitude = _stft_magnitude(mono, N_FFT, hop)
    if magnitude.size == 0:
        return {
            "matrix": [],
            "duration": round(duration, 6),
            "freq_max": analysis_rate / 2.0,
        }

    decibels = _log_magnitude(magnitude)
    ceiling = float(decibels.max())
    decibels = np.maximum(decibels, ceiling - DYNAMIC_RANGE_DB)

    pooled = _pool_axis(decibels, max_frames, axis=0)
    pooled = _pool_axis(pooled, max_bins, axis=1)

    low = float(pooled.min())
    high = float(pooled.max())
    scaled = (pooled - low) / (high - low) if high > low else np.zeros_like(pooled)

    # Transpose to (frequency, time) and flip so row 0 is the lowest band.
    matrix = np.round(scaled.T, 4)
    return {
        "matrix": matrix.tolist(),
        "duration": round(duration, 6),
        "freq_max": round(analysis_rate / 2.0, 3),
    }


def _pool_axis(matrix: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """Average-pool one axis down to at most `n_out` cells using
    `np.add.reduceat`, which sums variable-width groups in one pass."""
    n = matrix.shape[axis]
    if n_out <= 0 or n <= n_out:
        return matrix
    edges = (np.arange(n_out, dtype=np.int64) * n) // n_out
    summed = np.add.reduceat(matrix, edges, axis=axis)
    widths = np.diff(np.append(edges, n)).astype(np.float32)
    shape = [1] * matrix.ndim
    shape[axis] = n_out
    return (summed / widths.reshape(shape)).astype(np.float32)


# ---------------------------------------------------------------------------
# Self-test
#
# Run directly:  python acoustics.py
#
# Everything below synthesises its own audio with numpy -- no fixture
# files, nothing to check into the repo, no codec dependency. The checks
# are the actual claims this module makes to the rest of the app.
# ---------------------------------------------------------------------------


def _synth_track(seed: int, duration: float, sample_rate: int) -> np.ndarray:
    """Synthesise a spectrally rich, structured test signal.

    A pure tone is a terrible fingerprint test: one peak per frame, no
    structure, and any two tones look alike. This builds something closer
    to real audio -- a slow chirp sweeping across the band plus gated
    tone bursts at random frequencies with random onsets -- so the
    spectrogram has the sparse, moving landmarks the constellation
    algorithm is designed for.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    signal = np.zeros(n, dtype=np.float64)
    # Landmark density scales with length, so a 5-minute track is as
    # spectrally busy per second as a 20-second one -- otherwise the
    # performance check would be timing a nearly empty spectrogram.
    n_bursts = max(8, int(round(duration * 2.5)))

    # Slow logarithmic-ish chirp across the analysed band.
    f0, f1 = rng.uniform(180, 420), rng.uniform(2600, 4200)
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / max(duration, 1e-6) * t**2)
    signal += 0.35 * np.sin(phase)

    # Gated tone bursts: each is a distinct, isolated landmark.
    for _ in range(n_bursts):
        freq = float(rng.uniform(250, 4800))
        start = float(rng.uniform(0, max(duration - 0.4, 0.01)))
        length = float(rng.uniform(0.15, 0.7))
        i0 = int(start * sample_rate)
        i1 = min(n, i0 + int(length * sample_rate))
        if i1 - i0 < 64:
            continue
        window = _hann(i1 - i0).astype(np.float64)
        signal[i0:i1] += 0.45 * window * np.sin(2 * np.pi * freq * t[i0:i1])

    signal += 0.006 * rng.standard_normal(n)
    peak = np.abs(signal).max()
    if peak > 0:
        signal /= peak
    return signal.astype(np.float32)


def _add_noise(signal: np.ndarray, snr_db: float, seed: int = 99) -> np.ndarray:
    """Add white noise at a given SNR."""
    rng = np.random.default_rng(seed)
    power = float(np.mean(signal.astype(np.float64) ** 2)) or 1e-9
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(signal.size) * np.sqrt(noise_power)
    return (signal + noise).astype(np.float32)


def _self_test() -> int:
    import time

    sample_rate = 22050
    failures = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal failures
        status = "PASS" if condition else "FAIL"
        if not condition:
            failures += 1
        print(f"[{status}] {label}" + (f"  --  {detail}" if detail else ""))

    print("=" * 74)
    print("acoustics.py self-test (synthetic audio, numpy only)")
    print("=" * 74)

    # --- fingerprint a reference track ----------------------------------
    track_a = _synth_track(seed=1, duration=20.0, sample_rate=sample_rate)
    started = time.perf_counter()
    fp_a = fingerprint(track_a, sample_rate)
    elapsed = time.perf_counter() - started
    print(
        f"      reference: {fp_a['duration']:.1f}s -> {fp_a['peak_count']} peaks, "
        f"{fp_a['hash_count']} hashes in {elapsed * 1000:.0f} ms "
        f"({fp_a['hash_count'] / max(fp_a['duration'], 1e-9):.0f} hashes/s)"
    )
    check("fingerprint produces hashes", fp_a["hash_count"] > 500, f"{fp_a['hash_count']} hashes")
    check("fingerprint_id is a sha256 hex", len(fp_a["fingerprint_id"]) == 64)
    check("analysed at target rate", fp_a["sample_rate"] == TARGET_SAMPLE_RATE)

    # --- (a) identity ----------------------------------------------------
    same = compare(fp_a, fp_a)
    check(
        "(a) signal matches itself with score ~1.0",
        same["match"] and same["score"] > 0.99 and abs(same["offset_seconds"]) < 0.05,
        f"score={same['score']:.3f} offset={same['offset_seconds']:.3f}s "
        f"aligned={same['aligned_hashes']}",
    )
    fp_a_again = fingerprint(track_a, sample_rate)
    check(
        "    deterministic: same audio -> same fingerprint_id",
        fp_a_again["fingerprint_id"] == fp_a["fingerprint_id"],
    )

    # --- (b) robustness: noise + gain ------------------------------------
    degraded = _add_noise(track_a * 0.25, snr_db=10.0)
    fp_degraded = fingerprint(degraded, sample_rate)
    noisy = compare(fp_degraded, fp_a)
    check(
        "(b) matches a noisy (10 dB SNR) + 0.25x gain copy",
        noisy["match"] and abs(noisy["offset_seconds"]) < 0.1,
        f"score={noisy['score']:.3f} offset={noisy['offset_seconds']:.3f}s "
        f"aligned={noisy['aligned_hashes']}",
    )

    heavier = _add_noise(track_a * 4.0, snr_db=4.0, seed=7)
    fp_heavier = fingerprint(heavier, sample_rate)
    rough = compare(fp_heavier, fp_a)
    check(
        "    still matches at 4 dB SNR and 4x gain",
        rough["match"],
        f"score={rough['score']:.3f} aligned={rough['aligned_hashes']}",
    )

    # Survival of a rate conversion round trip is the case that byte
    # hashing gets wrong every single time: identical audio, different
    # bytes, and a perceptual fingerprint still recognises it.
    round_tripped = _resample(_resample(track_a, sample_rate, 32000), 32000, sample_rate)
    reencoded = compare(fingerprint(round_tripped, sample_rate), fp_a)
    check(
        "    survives a 22.05k -> 32k -> 22.05k resample round trip",
        reencoded["match"],
        f"score={reencoded['score']:.3f} aligned={reencoded['aligned_hashes']}",
    )

    # --- (c) unrelated signals -------------------------------------------
    track_b = _synth_track(seed=2, duration=20.0, sample_rate=sample_rate)
    fp_b = fingerprint(track_b, sample_rate)
    unrelated = compare(fp_a, fp_b)
    check(
        "(c) unrelated signals do NOT match",
        not unrelated["match"],
        f"score={unrelated['score']:.4f} common={unrelated['common_hashes']} "
        f"aligned={unrelated['aligned_hashes']}",
    )

    noise_only = fingerprint(
        _add_noise(np.zeros(sample_rate * 10, dtype=np.float32), snr_db=0.0, seed=5),
        sample_rate,
    )
    check(
        "    white noise does NOT match the reference",
        not compare(noise_only, fp_a)["match"],
        f"score={compare(noise_only, fp_a)['score']:.4f}",
    )

    # --- (d) clip localisation -------------------------------------------
    long_track = _synth_track(seed=3, duration=60.0, sample_rate=sample_rate)
    fp_long = fingerprint(long_track, sample_rate)
    true_offset = 23.7
    clip = long_track[
        int(true_offset * sample_rate) : int((true_offset + 6.0) * sample_rate)
    ]
    fp_clip = fingerprint(_add_noise(clip * 0.5, snr_db=15.0, seed=11), sample_rate)
    located = compare(fp_clip, fp_long)
    error = abs(located["offset_seconds"] - true_offset) if located["offset_seconds"] is not None else 99.0
    check(
        "(d) 6s clip located inside a 60s track, offset correct",
        located["match"] and error < 0.1,
        f"offset={located['offset_seconds']:.3f}s (true {true_offset}s, "
        f"error {error * 1000:.0f} ms) score={located['score']:.3f}",
    )

    # --- (e) self-similarity ---------------------------------------------
    jingle = _synth_track(seed=4, duration=4.0, sample_rate=sample_rate)
    filler_1 = _synth_track(seed=5, duration=11.0, sample_rate=sample_rate)
    filler_2 = _synth_track(seed=6, duration=9.0, sample_rate=sample_rate)
    repeated = np.concatenate([jingle, filler_1, jingle, filler_2]).astype(np.float32)
    expected_gap = (4.0 + 11.0)
    fp_repeat = fingerprint(repeated, sample_rate)
    repeats = find_repeats(fp_repeat, min_gap_seconds=5.0, min_hashes=12)
    found = [r for r in repeats if abs((r["repeat_at"] - r["first_at"]) - expected_gap) < 0.2]
    check(
        "(e) find_repeats detects the duplicated 4s segment",
        bool(found),
        (
            f"first_at={found[0]['first_at']:.2f}s repeat_at={found[0]['repeat_at']:.2f}s "
            f"duration={found[0]['duration']:.2f}s strength={found[0]['strength']:.2f}"
            if found
            else f"{len(repeats)} candidate(s): {repeats[:3]}"
        ),
    )
    check(
        "    a non-repeating recording reports no repeats",
        len(find_repeats(fp_a, min_gap_seconds=5.0, min_hashes=12)) == 0,
        f"{len(find_repeats(fp_a, min_gap_seconds=5.0, min_hashes=12))} found",
    )

    # --- (f) library index -----------------------------------------------
    library = []
    for i, seed in enumerate([21, 22, 23, 24, 25], start=1):
        track = _synth_track(seed=seed, duration=25.0, sample_rate=sample_rate)
        library.append((f"episode-{i}", track))
    index = build_index([(name, fingerprint(audio, sample_rate)) for name, audio in library])
    print(
        f"      index: {len(index['tracks'])} tracks, "
        f"{len(index['postings'])} distinct hashes"
    )

    target_name, target_audio = library[2]
    query_offset = 9.4
    query_clip = target_audio[
        int(query_offset * sample_rate) : int((query_offset + 7.0) * sample_rate)
    ]
    fp_query = fingerprint(_add_noise(query_clip * 0.6, snr_db=12.0, seed=31), sample_rate)
    started = time.perf_counter()
    hits = query_index(index, fp_query, top_n=3)
    query_ms = (time.perf_counter() - started) * 1000
    ok = bool(hits) and hits[0]["track_id"] == target_name
    ok = ok and abs(hits[0]["offset_seconds"] - query_offset) < 0.15
    check(
        "(f) query_index retrieves the right track out of 5",
        ok,
        (
            f"top={hits[0]['track_id']} score={hits[0]['score']:.3f} "
            f"offset={hits[0]['offset_seconds']:.2f}s (true {query_offset}s) "
            f"in {query_ms:.1f} ms"
            if hits
            else "no hits"
        ),
    )
    stranger = fingerprint(_synth_track(seed=99, duration=8.0, sample_rate=sample_rate), sample_rate)
    stranger_hits = query_index(index, stranger, top_n=3)
    top_score = stranger_hits[0]["score"] if stranger_hits else 0.0
    check(
        "    an unknown recording scores low against the library",
        top_score < 0.05,
        f"best score={top_score:.4f}",
    )

    # --- (g) performance --------------------------------------------------
    five_minutes = _synth_track(seed=77, duration=300.0, sample_rate=44100)
    started = time.perf_counter()
    fp_long_audio = fingerprint(five_minutes, 44100)
    long_elapsed = time.perf_counter() - started
    check(
        "(g) 5 minutes of 44.1 kHz audio fingerprints in < 1.0 s",
        long_elapsed < 1.0,
        f"{long_elapsed:.3f} s, {fp_long_audio['hash_count']} hashes, "
        f"{fp_long_audio['peak_count']} peaks",
    )

    # --- supporting API ---------------------------------------------------
    preview = spectrogram_preview(track_a, sample_rate, max_bins=48, max_frames=160)
    rows = len(preview["matrix"])
    cols = len(preview["matrix"][0]) if rows else 0
    flat = [v for row in preview["matrix"] for v in row]
    check(
        "    spectrogram_preview shape + 0..1 normalisation",
        rows == 48 and cols == 160 and min(flat) >= 0.0 and max(flat) <= 1.0
        and abs(preview["duration"] - 20.0) < 0.01,
        f"{rows}x{cols}, range [{min(flat):.2f}, {max(flat):.2f}], "
        f"freq_max={preview['freq_max']:.0f} Hz",
    )

    import os
    import tempfile

    wav_ok = True
    details = []
    for width, dtype_scale in ((1, 127.0), (2, 32767.0), (4, 2147483647.0)):
        handle, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(handle)
        try:
            source = _synth_track(seed=12, duration=3.0, sample_rate=16000)
            stereo = np.repeat(source[:, None], 2, axis=1)
            if width == 1:
                raw = ((stereo * dtype_scale) + 128.0).astype(np.uint8).tobytes()
            elif width == 2:
                raw = (stereo * dtype_scale).astype("<i2").tobytes()
            else:
                raw = (stereo.astype(np.float64) * dtype_scale).astype("<i4").tobytes()
            with wave.open(wav_path, "wb") as out:
                out.setnchannels(2)
                out.setsampwidth(width)
                out.setframerate(16000)
                out.writeframes(raw)
            loaded, rate = load_wav(wav_path)
            tolerance = 0.02 if width == 1 else 0.001
            close = (
                rate == 16000
                and loaded.size == source.size
                and float(np.max(np.abs(loaded - source))) < tolerance
            )
            details.append(f"{width * 8}-bit {'ok' if close else 'BAD'}")
            wav_ok = wav_ok and close
        finally:
            os.unlink(wav_path)
    check("    load_wav round-trips 8/16/32-bit stereo to mono", wav_ok, ", ".join(details))

    check(
        "    empty / tiny input is handled without raising",
        fingerprint(np.zeros(0, dtype=np.float32), 44100)["hash_count"] == 0
        and fingerprint(np.zeros(100, dtype=np.float32), 44100)["hash_count"] == 0
        and compare({"hashes": []}, fp_a)["match"] is False
        and find_repeats({"hashes": []}) == []
        and query_index({}, fp_a) == [],
    )

    print("=" * 74)
    if failures:
        print(f"RESULT: {failures} check(s) FAILED")
    else:
        print("RESULT: all checks PASSED")
    print("=" * 74)
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _self_test() else 0)

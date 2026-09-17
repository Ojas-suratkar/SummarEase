from app import create_app

app = create_app()

if __name__ == "__main__":
    # threaded=True: the app now does real concurrent work per request --
    # background summarization jobs poll from the browser while other
    # requests (the poll itself, the homepage, another tab) need to be
    # served at the same time. Flask's single-threaded dev-server default
    # would serialize all of that behind one request at a time.
    app.run(host="0.0.0.0", port=5000, debug=app.config["DEBUG"], threaded=True)

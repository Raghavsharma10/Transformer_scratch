def run_server(port=8000):
    """ Runs server on port with html response """
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class VerboseHTMLHandler(BaseHTTPRequestHandler):
        def do_HEAD(s):
            s.send_response(200)
            s.send_header("Content-type", "text/html")
            s.end_headers()

        def do_GET(s):
            global html

            data = changed_file()
            if data is not None:
                html = html_from_markdown(data)
            s.send_response(200)
            s.send_header("Content-type", "text/html")
            s.end_headers()
            s.wfile.write(standalone(html).encode('utf-8'))

    class SilentHTMLHandler(VerboseHTMLHandler):
        def log_message(self, format, *args):
            return

    port = int(port)
    server_class = HTTPServer
    handler = VerboseHTMLHandler if verbose else SilentHTMLHandler
    try:
        httpd = server_class(("localhost", port), handler)
    except PermissionError:
        sys.stderr.write("Permission denied\n")
        sys.exit(1)
    if verbose:
        print("Hosting server on port %d. Ctrl-c to exit" % port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    if verbose:
        print("\rShutting down server")
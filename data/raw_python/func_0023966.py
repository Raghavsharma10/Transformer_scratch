def _make_handler(state_token, done_function):
    '''
    Makes a a handler class to use inside the basic python HTTP server.

    state_token is the expected state token.
    done_function is a function that is called, with the code passed to it.
    '''

    class LocalServerHandler(BaseHTTPServer.BaseHTTPRequestHandler):

        def error_response(self, msg):
            logging.warn(
                'Error response: %(msg)s. %(path)s',
                msg=msg,
                path=self.path)
            self.send_response(400)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(msg)

        def do_GET(self):
            parsed = urlparse.urlparse(self.path)
            if len(parsed.query) == 0 or parsed.path != '/callback':
                self.error_response(
                    'We encountered a problem with your request.')
                return

            params = urlparse.parse_qs(parsed.query)
            if params['state'] != [state_token]:
                self.error_response(
                    'Attack detected: state tokens did not match!')
                return

            if len(params['code']) != 1:
                self.error_response('Wrong number of "code" query parameters.')
                return

            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(
                "courseraoauth2client: we have captured Coursera's response "
                "code. Feel free to close this browser window now and return "
                "to your terminal. Thanks!")
            done_function(params['code'][0])

    return LocalServerHandler
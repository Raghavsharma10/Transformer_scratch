def run_with_falcon(self):
        """
        runs the falcon/http based test server
        """
        from wsgiref import simple_server
        from zengine.server import app
        httpd = simple_server.make_server(self.manager.args.addr, int(self.manager.args.port), app)
        httpd.serve_forever()
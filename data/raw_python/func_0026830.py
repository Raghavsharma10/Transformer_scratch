def respond_static(self, environ):
        """
        Serves a static file when Django isn't being used.
        """
        path = os.path.normpath(environ["PATH_INFO"])
        if path == "/":
            content = self.index()
            content_type = "text/html"
        else:
            path = os.path.join(os.path.dirname(__file__), path.lstrip("/"))
            try:
                with open(path, "r") as f:
                    content = f.read()
            except IOError:
                return 404
            content_type = guess_type(path)[0]
        return (200, [("Content-Type", content_type)], content)
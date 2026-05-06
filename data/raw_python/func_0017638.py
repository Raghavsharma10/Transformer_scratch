def root(self, request, url):
        """
        Handles main URL routing for the databrowse app.

        `url` is the remainder of the URL -- e.g. 'comments/comment/'.
        """
        self.root_url = request.path[:len(request.path) - len(url)]
        url = url.rstrip('/')  # Trim trailing slash, if it exists.

        if url == '':
            return self.index(request)
        elif '/' in url:
            return self.model_page(request, *url.split('/', 2))

        raise http.Http404('The requested databrowse page does not exist.')
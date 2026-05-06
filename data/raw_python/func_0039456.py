def _sort_locations(locations):
        """
        Sort locations into "files" (archives) and "urls", and return
        a pair of lists (files,urls)
        """
        files = []
        urls = []

        # puts the url for the given file path into the appropriate
        # list
        def sort_path(path):
            url = path_to_url2(path)
            if mimetypes.guess_type(url, strict=False)[0] == 'text/html':
                urls.append(url)
            else:
                files.append(url)

        for url in locations:
            if url.startswith('file:'):
                path = url_to_path(url)
                if os.path.isdir(path):
                    path = os.path.realpath(path)
                    for item in os.listdir(path):
                        sort_path(os.path.join(path, item))
                elif os.path.isfile(path):
                    sort_path(path)
            else:
                urls.append(url)
        return files, urls
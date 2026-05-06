def create_request_with_query(self, kind, query, size="thumb", fmt="json"):
        """api/data.[fmt], api/images/[size].[fmt] api/files.[fmt]

        kind = ['data', 'images', 'files']


        """
        if kind == "data" or kind == "files":
            url = "{}/{}.{}".format(base_url, kind, fmt)
        elif kind == "images":
            url = "{}/images/{}.{}".format(base_url, size, fmt)
        self.url = url
        self.r = requests.get(url, params=unquote(urlencode(query)))
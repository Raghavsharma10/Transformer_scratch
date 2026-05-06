def create_articles(self, project, articleset, json_data=None, **options):
        """
        Create one or more articles in the set. Provide the needed arguments
        using the json_data or with key-value pairs
        @param json_data: A dictionary or list of dictionaries. Each dict
                          can contain a 'children' attribute which
                          is another list of dictionaries.
        """
        url = URL.article.format(**locals())
        # TODO duplicated from create_set, move into requests
        # (or separate post method?)
        if json_data is None:
            # form encoded request
            return self.request(url, method="post", data=options)
        else:
            if not isinstance(json_data, string_types):
                json_data = json.dumps(json_data, default=serialize)
            headers = {'content-type': 'application/json'}
            return self.request(url, method='post', data=json_data, headers=headers)
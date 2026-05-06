def create_set(self, project, json_data=None, **options):
        """
        Create a new article set. Provide the needed arguments using
        post_data or with key-value pairs
        """
        url = URL.articlesets.format(**locals())
        if json_data is None:
            # form encoded request
            return self.request(url, method="post", data=options)
        else:
            if not isinstance(json_data, (string_types)):
                json_data = json.dumps(json_data,default = serialize)
            headers = {'content-type': 'application/json'}
            return self.request(
                url, method='post', data=json_data, headers=headers)
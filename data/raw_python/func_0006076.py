def build_follow_url(host=None, **params):
        """
        Build a URL for the /follow page
        """

        # template = '?{params}'
        config = ExampleConfig()
        template = '/follow?{params}'

        if not host:
            host = config.get('example_web_hostname')

        return ExampleUrlBuilder.build(
            template=template,
            host=host,
            params=ExampleUrlBuilder.encode_params(**params)
        )
def set_applications_from_meta(self, metadata, exclude=None):
        """
        Parses meta and update or create each application
        :param str metadata: path or url to meta.yml
        :param list[str] exclude: List of application names, to exclude from meta.
                                  This might be need when you use meta as list of dependencies
        """
        if not exclude:
            exclude = []
        if metadata.startswith('http'):
            meta = yaml.safe_load(requests.get(url=metadata).content)
        else:
            # noinspection PyArgumentEqualDefault
            meta = yaml.safe_load(open(metadata, 'r').read())

        applications = []
        for app in meta['kit']['applications']:
            if app['name'] not in exclude:
                applications.append({
                    'name': app['name'],
                    'url': app['manifest']})
        self.restore({'applications': applications})
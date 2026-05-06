def route_path(self, path):
        '''
        Hacky function that's presently only useful for testing, gets the view
        that handles the given path.
        Later may be incorporated into the URL routing
        '''
        path = path.strip('/')
        name, _, subpath = path.partition('/')
        for service in singletons.settings.load_all('SERVICES'):
            if service.SERVICE_NAME == name:  # Found service!
                break
        else:
            return [], None  # found no service

        for partial_url, view in service.urls.items():
            partial_url = partial_url.strip('/')
            if isinstance(view, str):
                view = getattr(service, view)
            regexp = re.sub(r'<[^>]+>', r'([^/]+)', partial_url)
            matches = re.findall('^%s$' % regexp, subpath)
            if matches:
                if '(' not in regexp:
                    matches = []
                return matches, view

        return [], None
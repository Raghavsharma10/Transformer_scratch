def include(self, location, namespace=None, app_name=None):
        """
        Return an object suitable for url_patterns.

        :param location: root URL for all URLs from this router
        :param namespace: passed to url()
        :param app_name: passed to url()
        """
        sorted_entries = sorted(self.routes, key=operator.itemgetter(0),
                                reverse=True)

        arg = [u for _, u in sorted_entries]
        return url(location, urls.include(
            arg=arg,
            namespace=namespace,
            app_name=app_name))
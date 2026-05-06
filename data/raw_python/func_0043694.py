def url_for(self, resource, **kwargs):
        """Create a url for the given resource.

        :param resource: The resource
        :type resource: :class:`Resource`
        :param kwargs: Same arguments you would give :class:`flask.url_for`
        """
        if self.blueprint:
            return flask.url_for('.' + resource.endpoint, **kwargs)
        return flask.url_for(resource.endpoint, **kwargs)
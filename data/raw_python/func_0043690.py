def _register_view(self, app, resource, *urls, **kwargs):
        """Bind resources to the app.

        :param app: an actual :class:`flask.Flask` app
        :param resource:
        :param urls:

        :param endpoint: endpoint name (defaults to :meth:`Resource.__name__.lower`
            Can be used to reference this route in :meth:`flask.url_for`
        :type endpoint: str

        Additional keyword arguments not specified above will be passed as-is
        to :meth:`flask.Flask.add_url_rule`.

        SIDE EFFECT
            Implements the one mentioned in add_resource
        """
        endpoint = kwargs.pop('endpoint', None) or resource.__name__.lower()
        self.endpoints.add(endpoint)

        if endpoint in getattr(app, 'view_class', {}):
            existing_view_class = app.view_functions[endpoint].__dict__['view_class']

            # if you override the endpoint with a different class, avoid the collision by raising an exception
            if existing_view_class != resource:
                raise ValueError('Endpoint {!r} is already set to {!r}.'
                                 .format(endpoint, existing_view_class.__name__))

        if not hasattr(resource, 'endpoint'):  # Don't replace existing endpoint
            resource.endpoint = endpoint
        resource_func = self.output(resource.as_view(endpoint))

        for decorator in chain(kwargs.pop('decorators', ()), self.decorators):
            resource_func = decorator(resource_func)

        for url in urls:
            rule = self._make_url(url, self.blueprint.url_prefix if self.blueprint else None)

            # If this Api has a blueprint
            if self.blueprint:
                # And this Api has been setup
                if self.blueprint_setup:
                    # Set the rule to a string directly, as the blueprint
                    # is already set up.
                    self.blueprint_setup.add_url_rule(self._make_url(url, None), view_func=resource_func, **kwargs)
                    continue
                else:
                    # Set the rule to a function that expects the blueprint
                    # prefix to construct the final url.  Allows deferment
                    # of url finalization in the case that the Blueprint
                    # has not yet been registered to an application, so we
                    # can wait for the registration prefix
                    rule = partial(self._make_url, url)
            else:
                # If we've got no Blueprint, just build a url with no prefix
                rule = self._make_url(url, None)
            # Add the url to the application or blueprint
            app.add_url_rule(rule, view_func=resource_func, **kwargs)
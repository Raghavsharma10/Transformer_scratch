def bind(self, flask_app, service, group=None):
        """Bind the service API urls to a flask app."""
        if group not in self.services[service]:
            raise RuntimeError(
                'API group {} does not exist in service {}'.format(
                    group, service)
            )
        for name, api in self.services[service][group].items():
            # only bind APIs that have views associated with them
            if api.view_fn is None:
                continue
            if name not in flask_app.view_functions:
                flask_app.add_url_rule(
                    api.url, name, view_func=api.view_fn, **api.options)
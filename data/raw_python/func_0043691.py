def _add_url_rule_patch(blueprint_setup, rule, endpoint=None, view_func=None, **options):
        """Patch BlueprintSetupState.add_url_rule for delayed creation.

        Method used for setup state instance corresponding to this Api
        instance.  Exists primarily to enable _make_url's function.

        :param blueprint_setup: The BlueprintSetupState instance (self)
        :param rule: A string or callable that takes a string and returns a
            string(_make_url) that is the url rule for the endpoint
            being registered
        :param endpoint: See :meth:`flask.BlueprintSetupState.add_url_rule`
        :param view_func: See :meth:`flask.BlueprintSetupState.add_url_rule`
        :param **options: See :meth:`flask.BlueprintSetupState.add_url_rule`
        """
        if callable(rule):
            rule = rule(blueprint_setup.url_prefix)
        elif blueprint_setup.url_prefix:
            rule = blueprint_setup.url_prefix + rule
        options.setdefault('subdomain', blueprint_setup.subdomain)
        if endpoint is None:
            endpoint = _endpoint_from_view_func(view_func)
        defaults = blueprint_setup.url_defaults
        if 'defaults' in options:
            defaults = dict(defaults, **options.pop('defaults'))
        blueprint_setup.app.add_url_rule(rule, '%s.%s' % (blueprint_setup.blueprint.name, endpoint),
                                         view_func, defaults=defaults, **options)
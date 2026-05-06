def route_as_view(cls, app, name, rules, *class_args, **class_kwargs):
        """ Register the view with an URL route
        :param app: Flask application
        :type app: flask.Flask|flask.Blueprint
        :param name: Unique view name
        :type name: str
        :param rules: List of route rules to use
        :type rules: Iterable[str|werkzeug.routing.Rule]
        :param class_args: Args to pass to object constructor
        :param class_kwargs: KwArgs to pass to object constructor
        :return: View callable
        :rtype: Callable
        """
        view = super(MethodView, cls).as_view(name, *class_args, **class_kwargs)
        for rule in rules:
            app.add_url_rule(rule, view_func=view)
        return view
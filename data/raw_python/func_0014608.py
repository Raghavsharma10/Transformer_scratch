def route(app_or_blueprint, rule, **options):
    """An alternative to :meth:`flask.Flask.route` or :meth:`flask.Blueprint.route` that
    always adds the ``POST`` method to the allowed endpoint request methods.

    You should use this for all your view functions that would need to use Sijax.

    We're doing this because Sijax uses ``POST`` for data passing,
    which means that every endpoint that wants Sijax support
    would have to accept ``POST`` requests.

    Registering functions that would use Sijax should happen like this::

        @flask_sijax.route(app, '/')
        def index():
            pass

    If you remember to make your view functions accessible via POST
    like this, you can avoid using this decorator::

        @app.route('/', methods=['GET', 'POST'])
        def index():
            pass
    """
    def decorator(f):
        methods = options.pop('methods', ('GET', 'POST'))
        if 'POST' not in methods:
            methods = tuple(methods) + ('POST',)
        options['methods'] = methods
        app_or_blueprint.add_url_rule(rule, None, f, **options)
        return f
    return decorator
def gunicorn(host, port, workers):
    """use gunicorn"""
    from gunicorn.app.base import Application

    class FlaskApplication(Application):
        def init(self, parser, opts, args):
            return {'bind': '{0}:{1}'.format(host, port), 'workers': workers}

        def load(self):
            return app

    application = FlaskApplication()
    return application.run()
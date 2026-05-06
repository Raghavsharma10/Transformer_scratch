def routes(cls, application=None):
        """
        Method for adding the routes to the `tornado.web.Application`.
        """
        if application:
            for route in cls._routes:
                application.add_handlers(route['host'], route['spec'])
        else:
            return [route['spec'] for route in cls._routes]
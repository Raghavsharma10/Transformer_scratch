def route(cls, route, config=None):
        """
            This method provides a decorator for adding endpoints to the
            http server.

            Args:
                route (str): The url to be handled by the RequestHandled
                config (dict): Configuration for the request handler

            Example:

                .. code-block:: python

                    import nautilus
                    from nauilus.network.http import RequestHandler

                    class MyService(nautilus.Service):
                        # ...

                    @MyService.route('/')
                    class HelloWorld(RequestHandler):
                        def get(self):
                            return self.finish('hello world')
        """
        def decorator(wrapped_class, **kwds):

            # add the endpoint at the given route
            cls._routes.append(
                dict(url=route, request_handler=wrapped_class)
            )
            # return the class undecorated
            return wrapped_class

        # return the decorator
        return decorator
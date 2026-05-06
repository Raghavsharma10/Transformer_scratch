def run_on_main_thread(self, func, args=None, kwargs=None):
        """
        Runs the ``func`` callable on the main thread, by using the provided microservice
        instance's IOLoop.

        :param func: callable to run on the main thread
        :param args: tuple or list with the positional arguments.
        :param kwargs: dict with the keyword arguments.
        :return:
        """
        if not args:
            args = ()
        if not kwargs:
            kwargs = {}
        self.microservice.get_io_loop().add_callback(func, *args, **kwargs)
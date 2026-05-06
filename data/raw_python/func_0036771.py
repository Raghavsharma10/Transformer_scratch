def launch_thread(self, name, fn, *args, **kwargs):
        """
        Adds a named thread to the "thread pool" dictionary of Thread objects.

        A daemon thread that executes the passed-in function `fn` with the
        given args and keyword args is started and tracked in the `thread_pool`
        attribute with the given `name` as the key.
        """
        logger.debug(
            "Launching thread '%s': %s(%s, %s)", name,
            fn, args, kwargs
        )
        self.thread_pool[name] = threading.Thread(
            target=fn, args=args, kwargs=kwargs
        )
        self.thread_pool[name].daemon = True
        self.thread_pool[name].start()
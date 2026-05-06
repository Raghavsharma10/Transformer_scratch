def run(self, target, args=()):
        """ Run a function in a separate thread.

        :param target: the function to run.
        :param args: the parameters to pass to the function.
        """
        run_event = threading.Event()
        run_event.set()
        thread = threading.Thread(target=target, args=args + (run_event, ))
        self.thread_pool.append(thread)
        self.run_events.append(run_event)
        thread.start()
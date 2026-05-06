def start_thread(self, target, args, kwargs):
        """
        Shortcut method for starting a thread.

        :param target: The function to be executed.
        :param args: A tuple or list representing the positional arguments for the thread.
        :param kwargs: A dictionary representing the keyword arguments.

        .. versionadded:: 0.5.0
        """
        thread_obj = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
        thread_obj.start()
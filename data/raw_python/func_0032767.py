def promise(cls, fn, *args, **kwargs):
        """
        Used to build a task based on a callable function and the arguments.
        Kick it off and start execution of the task.

        :param fn: callable
        :param args: tuple
        :param kwargs: dict
        :return: SynchronousTask or AsynchronousTask
        """
        task = cls.task(target=fn, args=args, kwargs=kwargs)
        task.start()
        return task
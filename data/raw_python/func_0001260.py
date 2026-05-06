def add_task(self, task, func=None, **kwargs):
        ''' Add a task parser '''
        if not self.__tasks:
            raise Exception("Tasks subparsers is disabled")
        if 'help' not in kwargs:
            if func.__doc__:
                kwargs['help'] = func.__doc__
        task_parser = self.__tasks.add_parser(task, **kwargs)
        if self.__add_vq:
            self.add_vq(task_parser)
        if func is not None:
            task_parser.set_defaults(func=func)
        return task_parser
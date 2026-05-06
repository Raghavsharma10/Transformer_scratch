def execute(self):
        """
        Invoke the redispy pipeline.execute() method and take all the values
        returned in sequential order of commands and map them to the
        Future objects we returned when each command was queued inside
        the pipeline.
        Also invoke all the callback functions queued up.
        :param raise_on_error: boolean
        :return: None
        """
        stack = self._stack
        callbacks = self._callbacks

        promises = []
        if stack:
            def process():
                """
                take all the commands and pass them to redis.
                this closure has the context of the stack
                :return: None
                """

                # get the connection to redis
                pipe = ConnectionManager.get(self.connection_name)

                # keep track of all the commands
                call_stack = []

                # build a corresponding list of the futures
                futures = []

                # we need to do this because we need to make sure
                # all of these are callable.
                # there shouldn't be any non-callables.
                for item, args, kwargs, future in stack:
                    f = getattr(pipe, item)
                    if callable(f):
                        futures.append(future)
                        call_stack.append((f, args, kwargs))

                # here's where we actually pass the commands to the
                # underlying redis-py pipeline() object.
                for f, args, kwargs in call_stack:
                    f(*args, **kwargs)

                # execute the redis-py pipeline.
                # map all of the results into the futures.
                for i, v in enumerate(pipe.execute()):
                    futures[i].set(v)

            promises.append(process)

        # collect all the other pipelines for other named connections attached.
        promises += [p.execute for p in self._pipelines.values()]
        if len(promises) == 1:
            promises[0]()
        else:
            # if there are no promises, this is basically a no-op.
            TaskManager.wait(*[TaskManager.promise(p) for p in promises])

        for cb in callbacks:
            cb()
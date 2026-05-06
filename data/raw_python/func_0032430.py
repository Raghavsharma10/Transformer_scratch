def reset(self):
        """
        cleanup method. get rid of the stack and callbacks.
        :return:
        """
        self._stack = []
        self._callbacks = []
        pipes = self._pipelines
        self._pipelines = {}
        for pipe in pipes.values():
            pipe.reset()
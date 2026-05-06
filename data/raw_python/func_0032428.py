def _pipeline(self, name):
        """
        Don't call this function directly.
        Used by the NestedPipeline class when it executes.
        :param name:
        :return:
        """
        if name == self.connection_name:
            return self

        try:
            return self._pipelines[name]
        except KeyError:
            pipe = Pipeline(name=name, autoexec=True)
            self._pipelines[name] = pipe
            return pipe
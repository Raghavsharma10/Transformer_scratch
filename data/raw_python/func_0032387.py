def connect(cls, pipeline_method, name=None):
        """
        Low level logic to bind a callable method to a name.
        Don't call this directly unless you know what you are doing.

        :param pipeline_method: callable
        :param name: str optional
        :return: None
        """
        new_pool = pipeline_method().connection_pool
        try:
            if cls.get(name).connection_pool != new_pool:
                raise AlreadyConnected("can't change connection for %s" % name)
        except InvalidPipeline:
            pass

        cls.connections[name] = pipeline_method
def get_instance_of(self, model_cls):
        """
        Search the data to find a instance
        of a model specified in the template
        """
        for obj in self.data.values():
            if isinstance(obj, model_cls):
                return obj
        LOGGER.error('Context Not Found')
        raise Exception('Context Not Found')
def get_context(self):
        """
        Create a dict with the context data
        context is not required, but if it
        is defined it should be a tuple
        """
        if not self.context:
            return
        else:
            assert isinstance(self.context, tuple), 'Expected a Tuple not {0}'.format(type(self.context))
        for model in self.context:
            model_cls = utils.get_model_class(model)
            key = utils.camel_to_snake(model_cls.__name__)
            self.context_data[key] = self.get_instance_of(model_cls)
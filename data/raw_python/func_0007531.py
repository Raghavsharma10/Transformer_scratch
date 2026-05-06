def get_dict(self, obj, state=None, base_name='View'):
        """The style dict for a view instance.

        """
        return self.get_dict_for_class(class_name=obj.__class__,
                                       state=obj.state,
                                       base_name=base_name)
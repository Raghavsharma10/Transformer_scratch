def get_param_type(self, index):
        """Returns a ReturnValue instance for param type 'index'"""

        assert index in (0, 1)

        type_info = self.type.get_param_type(index)
        type_cls = get_return_class(type_info)
        instance = type_cls(None, type_info, [], self.backend)
        instance.setup()
        return instance
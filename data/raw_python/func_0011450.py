def _wrap_type_instantiation(self, type_cls):
        """Wrap the creation of the type so that we can provide
        a null-stream to initialize it"""
        def wrapper(*args, **kwargs):
            # use args for struct arguments??
            return type_cls(stream=self._null_stream)
        return wrapper
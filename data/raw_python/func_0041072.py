def to_dict(self, include_args=True, include_kwargs=True):
        """Converts this object to a dictionary.

        :param include_args: boolean indicating whether to include the
                             exception args in the output.
        :param include_kwargs: boolean indicating whether to include the
                               exception kwargs in the output.
        """
        data = {
            'exception_str': self.exception_str,
            'traceback_str': self.traceback_str,
            'exc_type_names': self.exception_type_names,
            'exc_args': self.exception_args if include_args else tuple(),
            'exc_kwargs': self.exception_kwargs if include_kwargs else {},
            'generated_on': self.generated_on,
        }
        if self._cause is not None:
            data['cause'] = self._cause.to_dict(include_args=include_args,
                                                include_kwargs=include_kwargs)
        return data
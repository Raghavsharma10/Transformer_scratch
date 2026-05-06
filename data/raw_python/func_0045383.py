def _write_field(self, value):
        """
        Write a single field to the destination file.

        :param T value: The value of the field.
        """
        class_name = str(value.__class__)
        if class_name not in self.handlers:
            raise ValueError('No handler has been registered for class: {0!s}'.format(class_name))
        handler = self.handlers[class_name]
        handler(value, self._file)
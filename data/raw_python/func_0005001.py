def register(self, field, shape, dtype):
        '''Register a field as a tensor with specified shape and type.

        A `Tensor` of the given shape and type will be registered in this
        object's `fields` dict.

        Parameters
        ----------
        field : str
            The name of the field

        shape : iterable of `int` or `None`
            The shape of the output variable.
            This does not include a dimension for multiple outputs.

            `None` may be used to indicate variable-length outputs

        dtype : type
            The data type of the field

        Raises
        ------
        ParameterError
            If dtype or shape are improperly specified
        '''
        if not isinstance(dtype, type):
            raise ParameterError('dtype={} must be a type'.format(dtype))

        if not (isinstance(shape, Iterable) and
                all([s is None or isinstance(s, int) for s in shape])):
            raise ParameterError('shape={} must be an iterable of integers'.format(shape))

        self.fields[self.scope(field)] = Tensor(tuple(shape), dtype)
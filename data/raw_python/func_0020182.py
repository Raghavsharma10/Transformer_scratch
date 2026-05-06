def get_value(self, instance, *bits):
        '''Retrieve the value :class:`Field` from a :class:`StdModel`
``instance``.

:param instance: The :class:`StdModel` ``instance`` invoking this function.
:param bits: Additional information for nested fields which derives from
    the :ref:`double underscore <tutorial-underscore>` notation.
:return: the value of this :class:`Field` in the ``instance``. can raise
    :class:`AttributeError`.

This method is used by the :meth:`StdModel.get_attr_value` method when
retrieving values form a :class:`StdModel` instance.
'''
        if bits:
            raise AttributeError
        else:
            return getattr(instance, self.attname)
def any_ref(self, *, column, value=missing, dumped=False, inner=False):
        """Returns a NamedTuple of (name, type, value) for any type of reference.

        .. code-block:: python

            # Name ref
            >>> tracker.any_ref(column=User.email)
            Reference(name='email', type='name', value=None)

            # Value ref
            >>> tracker.any_ref(column=User.email, value='user@domain')
            Reference(name='email', type='value', value={'S': 'user@domain'})

            # Passed as value ref, but value is another column
            >>> tracker.any_ref(column=User.email, value=User.other_column)
            Reference(name='other_column', type='name', value=None)

        :param column: The column to reference.  If ``value`` is None, this will render a name ref for this column.
        :type column: :class:`~bloop.conditions.ComparisonMixin`
        :param value: *(Optional)* If provided, this is likely a value ref.  If ``value`` is also a column,
            this will render a name ref for that column (not the ``column`` parameter).
        :param bool dumped:  *(Optional)* True if the value has already been dumped and should not be dumped
            through the column's typedef again.  Commonly used with atomic conditions (which store the object's dumped
            representation).  Default is False.
        :param bool inner: *(Optional)* True if this is a value ref and it should be dumped through a collection's
            inner type, and not the collection type itself.  Default is False.
        :return: A name or value reference
        :rtype: :class:`bloop.conditions.Reference`
        """
        # Can't use None since it's a legal value for comparisons (attribute_not_exists)
        if value is missing:
            # Simple path ref to the column.
            name = self._path_ref(column=column)
            ref_type = "name"
            value = None
        elif isinstance(value, ComparisonMixin):
            # value is also a column!  Also a path ref.
            name = self._path_ref(column=value)
            ref_type = "name"
            value = None
        else:
            # Simple value ref.
            name, value = self._value_ref(column=column, value=value, dumped=dumped, inner=inner)
            ref_type = "value"
        return Reference(name=name, type=ref_type, value=value)
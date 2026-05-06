def set_cumulative(self, cumulative):
        """Applies this rule to all previous assessment parts.

        arg:    cumulative (boolean): ``true`` to apply to all previous
                assessment parts. ``false`` to apply to the immediate
                previous assessment part
        raise:  InvalidArgument - ``cumulative`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.set_group_template
        if self.get_cumulative_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_boolean(cumulative):
            raise errors.InvalidArgument()
        self._my_map['cumulative'] = cumulative
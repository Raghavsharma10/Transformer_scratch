def set_distribute_compositions(self, distribute_comps):
        """Sets the distribution rights.

        This sets distribute verbatim to ``true``.

        arg:    distribute_comps (boolean): right to distribute
                modifications
        raise:  InvalidArgument - ``distribute_comps`` is invalid
        raise:  NoAccess - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.set_group_template
        if self.get_distribute_compositions_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_boolean(distribute_comps):
            raise errors.InvalidArgument()
        self._my_map['distributeCompositions'] = distribute_comps
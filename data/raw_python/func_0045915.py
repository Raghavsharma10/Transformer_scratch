def set_distribute_verbatim(self, distribute_verbatim):
        """Sets the distribution rights.

        arg:    distribute_verbatim (boolean): right to distribute
                verbatim copies
        raise:  InvalidArgument - ``distribute_verbatim`` is invalid
        raise:  NoAccess - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.set_group_template
        if self.get_distribute_verbatim_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_boolean(distribute_verbatim):
            raise errors.InvalidArgument()
        self._my_map['distributeVerbatim'] = distribute_verbatim
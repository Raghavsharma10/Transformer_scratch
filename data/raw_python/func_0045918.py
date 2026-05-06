def set_distribute_alterations(self, distribute_mods):
        """Sets the distribute alterations flag.

        This also sets distribute verbatim to ``true``.

        arg:    distribute_mods (boolean): right to distribute
                modifications
        raise:  InvalidArgument - ``distribute_mods`` is invalid
        raise:  NoAccess - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.set_group_template
        if self.get_distribute_alterations_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_boolean(distribute_mods):
            raise errors.InvalidArgument()
        self._my_map['distributeAlterations'] = distribute_mods
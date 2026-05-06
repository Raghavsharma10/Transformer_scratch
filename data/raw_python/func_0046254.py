def set_distribute_alterations(self, distribute_mods=None):
        """Sets the distribute alterations flag.

        This also sets distribute verbatim to ``true``.

        :param distribute_mods: right to distribute modifications
        :type distribute_mods: ``boolean``
        :raise: ``InvalidArgument`` -- ``distribute_mods`` is invalid
        :raise: ``NoAccess`` -- authorization failure

        *compliance: mandatory -- This method must be implemented.*

        """
        if distribute_mods is None:
            raise NullArgument()
        metadata = Metadata(**settings.METADATA['distribute_alterations'])
        if metadata.is_read_only():
            raise NoAccess()
        if self._is_valid_input(distribute_mods, metadata, array=False):
            self._my_map['canDistributeAlterations'] = distribute_mods
        else:
            raise InvalidArgument()
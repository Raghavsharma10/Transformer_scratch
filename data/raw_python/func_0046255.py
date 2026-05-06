def set_distribute_compositions(self, distribute_comps=None):
        """Sets the distribution rights.

        This sets distribute verbatim to ``true``.

        :param distribute_comps: right to distribute modifications
        :type distribute_comps: ``boolean``
        :raise: ``InvalidArgument`` -- ``distribute_comps`` is invalid
        :raise: ``NoAccess`` -- authorization failure

        *compliance: mandatory -- This method must be implemented.*

        """
        if distribute_comps is None:
            raise NullArgument()
        metadata = Metadata(**settings.METADATA['distribute_compositions'])
        if metadata.is_read_only():
            raise NoAccess()
        if self._is_valid_input(distribute_comps, metadata, array=False):
            self._my_map['canDistributeCompositions'] = distribute_comps
        else:
            raise InvalidArgument()
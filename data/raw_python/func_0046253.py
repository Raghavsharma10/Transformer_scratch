def set_distribute_verbatim(self, distribute_verbatim=None):
        """Sets the distribution rights.

        :param distribute_verbatim: right to distribute verbatim copies
        :type distribute_verbatim: ``boolean``
        :raise: ``InvalidArgument`` -- ``distribute_verbatim`` is invalid
        :raise: ``NoAccess`` -- authorization failure

        *compliance: mandatory -- This method must be implemented.*

        """
        if distribute_verbatim is None:
            raise NullArgument()
        metadata = Metadata(**settings.METADATA['distribute_verbatim'])
        if metadata.is_read_only():
            raise NoAccess()
        if self._is_valid_input(distribute_verbatim, metadata, array=False):
            self._my_map['canDistributeVerbatim'] = distribute_verbatim
        else:
            raise InvalidArgument()
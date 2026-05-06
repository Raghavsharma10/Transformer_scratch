def set_principal_credit_string(self, credit_string=None):
        """Sets the principal credit string.

        :param credit_string: the new credit string
        :type credit_string: ``string``
        :raise: ``InvalidArgument`` -- ``credit_string`` is invalid
        :raise: ``NoAccess`` -- ``Metadata.isReadOnly()`` is ``true``
        :raise: ``NullArgument`` -- ``credit_string`` is ``null``

        *compliance: mandatory -- This method must be implemented.*

        """
        if credit_string is None:
            raise NullArgument()
        metadata = Metadata(**settings.METADATA['principal_credit_string'])
        if metadata.is_read_only():
            raise NoAccess()
        if self._is_valid_input(credit_string, metadata, array=False):
            self._my_map['principalCreditString']['text'] = credit_string
        else:
            raise InvalidArgument()
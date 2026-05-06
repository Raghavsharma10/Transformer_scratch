def get_response_record(self, item_record_type):
        """Gets the response record corresponding to the given ``Item`` record ``Type``.

        This method is used to retrieve an object implementing the
        requested record. The ``item_record_type`` may be the ``Type``
        returned in ``get_record_types()`` or any of its parents in a
        ``Type`` hierarchy where ``has_record_type(item_record_type)``
        is ``true`` .

        arg:    item_record_type (osid.type.Type): an item record type
        return: (osid.assessment.records.ResponseRecord) - the response
                record
        raise:  NullArgument - ``item_record_type`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unsupported - ``has_record_type(item_record_type)`` is
                ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        if not self.has_record_type(item_record_type):
            raise errors.Unsupported()
        if str(item_record_type) not in self._records:
            raise errors.Unimplemented()
        return self._records[str(item_record_type)]
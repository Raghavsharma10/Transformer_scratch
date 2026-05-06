def get_next_types(self, n=None):
        """Gets the next set of ``Types`` in this list.

        The specified amount must be less than or equal to the return
        from ``available()``.

        arg:    n (cardinal): the number of ``Type`` elements requested
                which must be less than or equal to ``available()``
        return: (osid.type.Type) - an array of ``Type`` elements.The
                length of the array is less than or equal to the number
                specified.
        raise:  IllegalState - no more elements available in this list
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        if n > self.available():
            # !!! This is not quite as specified (see method docs) !!!
            raise IllegalState('not enough elements available in this list')
        else:
            next_list = []
            i = 0
            while i < n:
                try:
                    next_list.append(next(self))
                except:  # Need to specify exceptions here
                    raise OperationFailed()
                i += 1
            return next_list
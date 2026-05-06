def get_next_parameters(self, n=None):
        """Gets the next set of ``Parameters`` in this list which must be less than or equal to the return from ``available()``.

        arg:    n (cardinal): the number of ``Parameter`` elements
                requested which must be less than or equal to
                ``available()``
        return: (osid.configuration.Parameter) - an array of
                ``Parameter`` elements.The length of the array is less
                than or equal to the number specified.
        raise:  IllegalState - no more elements available in this list
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceList.get_next_resources
        if n > self.available():
            # !!! This is not quite as specified (see method docs) !!!
            raise IllegalState('not enough elements available in this list')
        else:
            next_list = []
            x = 0
            while x < n:
                try:
                    next_list.append(self.next())
                except:  # Need to specify exceptions here
                    raise OperationFailed()
                x = x + 1
            return next_list
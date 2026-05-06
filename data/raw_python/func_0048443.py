def get_next_id(self):
        """Gets the next Id in this list.

        return: (osid.id.Id) - the next Id in this list. The has_next()
                method should be used to test that a next Id is
                available before calling this method.
        raise:  IllegalState - no more elements available in this list
        raise:  OperationFailed - unable to complete request
        compliance: mandatory - This method must be implemented.

        """
        try:
            next_item = next(self)
        except StopIteration:
            raise IllegalState('no more elements available in this list')
        except Exception:  # Need to specify exceptions here!
            raise OperationFailed()
        else:
            return next_item
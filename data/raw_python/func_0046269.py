def get_next_repository(self):
        """Gets the next ``Repository`` in this list.

        :return: the next ``Repository`` in this list. The ``has_next()`` method should be used to test that a next ``Repository`` is available before calling this method.
        :rtype: ``osid.repository.Repository``
        :raise: ``IllegalState`` -- no more elements available in this list
        :raise: ``OperationFailed`` -- unable to complete request

        *compliance: mandatory -- This method must be implemented.*

        """
        try:
            next_object = next(self)
        except StopIteration:
            raise IllegalState('no more elements available in this list')
        except Exception:  # Need to specify exceptions here!
            raise OperationFailed()
        else:
            return next_object
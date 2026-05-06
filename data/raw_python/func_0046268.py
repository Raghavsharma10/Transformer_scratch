def get_next_asset_content(self):
        """Gets the next AssetContent in this list.

        return: (osid.repository.AssetContent) - the next AssetContent
                in this list. The has_next() method should be used to
                test that a next AssetContent is available before
                calling this method.
        raise:  IllegalState - no more elements available in this list
        raise:  OperationFailed - unable to complete request
        compliance: mandatory - This method must be implemented.

        """
        try:
            next_object = next(self)
        except StopIteration:
            raise IllegalState('no more elements available in this list')
        except Exception:  # Need to specify exceptions here!
            raise OperationFailed()
        else:
            return next_object
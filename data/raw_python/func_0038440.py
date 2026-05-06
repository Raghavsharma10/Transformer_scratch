def _get_id(self, rdata):
        """
        Returns jsonrpc request's id value or None if there is none.

        InvalidRequestError will be raised if the id value has invalid type.
        """
        if 'id' in rdata:
            if isinstance(rdata['id'], basestring) or \
                    isinstance(rdata['id'], int) or \
                    isinstance(rdata['id'], long) or \
                    isinstance(rdata['id'], float) or \
                    rdata['id'] is None:
                return rdata['id']
            else:
                # invalid type
                raise InvalidRequestError
        else:
            # It's a notification.
            return None
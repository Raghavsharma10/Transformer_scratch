def _get_params(self, rdata):
        """
        Returns a list of jsonrpc request's method parameters.
        """
        if 'params' in rdata:
            if isinstance(rdata['params'], dict) \
                    or isinstance(rdata['params'], list) \
                    or rdata['params'] is None:
                return rdata['params']
            else:
                # wrong type
                raise InvalidRequestError
        else:
            return None
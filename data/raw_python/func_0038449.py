def dumps(self):
        """Return the Exception data in a format for JSON-RPC."""

        error = {'code': self.code,
                 'message': str(self.message)}

        if self.data is not None:
            error['data'] = self.data

        return error
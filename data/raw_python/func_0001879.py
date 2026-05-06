def get(self, uid=None):
        """Example retrieve API method.
        """
        # Return resource collection

        if uid is None:
            return self.response_factory.ok(data=resource_db)

        # Return resource based on UID.

        try:
            record = [r for r in resource_db if r.get('id') == uid].pop()

        except IndexError:
            return self.response_factory.not_found(errors=['Resource with UID {} does not exist.'.format(uid)])

        return self.response_factory.ok(data=record)
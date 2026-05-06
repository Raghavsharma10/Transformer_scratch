def delete(self, uid):
        """Example DELETE method.
        """
        try:
            record = resource_db[uid].copy()

        except KeyError:
            return self.response_factory.not_found(errors=['Resource with UID {} does not exist!'])

        del resource_db[uid]

        return self.response_factory.ok(data=record)
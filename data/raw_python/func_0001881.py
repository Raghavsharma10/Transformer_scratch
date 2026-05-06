def put(self, uid):
        """Example PUT method.
        """

        resource_data = self.request.json

        try:
            record = resource_db[uid]

        except KeyError:
            return self.response_factory.not_found(errors=['Resource with UID {} does not exist!'])

        record['name'] = resource_data.get('name')

        return self.response_factory.ok(data=record)
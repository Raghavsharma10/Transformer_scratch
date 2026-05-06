def post(self):
        """Example POST method.
        """

        resource_data = self.request.json

        record = {'id': str(len(resource_db) + 1),
                  'name': resource_data.get('name')}

        resource_db.append(record)

        return self.response_factory.ok(data=record)
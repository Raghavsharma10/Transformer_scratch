def change(self, event):
        """Change an existing object"""

        try:
            data, schema, user, client = self._get_args(event)
        except AttributeError:
            return

        try:
            uuid = data['uuid']
            change = data['change']
            field = change['field']
            new_data = change['value']
        except KeyError as e:
            self.log("Update request with missing arguments!", data, e,
                     lvl=critical)
            self._cancel_by_error(event, 'missing_args')
            return

        storage_object = None

        try:
            storage_object = objectmodels[schema].find_one({'uuid': uuid})
        except Exception as e:
            self.log('Change for unknown object requested:', schema, data, lvl=warn)

        if storage_object is None:
            self._cancel_by_error(event, 'not_found')
            return

        if not self._check_permissions(user, 'write', storage_object):
            self._cancel_by_permission(schema, data, event)
            return

        self.log("Changing object:", storage_object._fields, lvl=debug)
        storage_object._fields[field] = new_data

        self.log("Storing object:", storage_object._fields, lvl=debug)
        try:
            storage_object.validate()
        except ValidationError:
            self.log("Validation of changed object failed!",
                     storage_object, lvl=warn)
            self._cancel_by_error(event, 'invalid_object')
            return

        storage_object.save()

        self.log("Object stored.")

        result = {
            'component': 'hfos.events.objectmanager',
            'action': 'change',
            'data': {
                'schema': schema,
                'uuid': uuid
            }
        }

        self._respond(None, result, event)
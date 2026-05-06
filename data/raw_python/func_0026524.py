def get(self, event):
        """Get a specified object"""

        try:
            data, schema, user, client = self._get_args(event)
        except AttributeError:
            return

        object_filter = self._get_filter(event)

        if 'subscribe' in data:
            do_subscribe = data['subscribe'] is True
        else:
            do_subscribe = False

        try:
            uuid = str(data['uuid'])
        except (KeyError, TypeError):
            uuid = ""

        opts = schemastore[schema].get('options', {})
        hidden = opts.get('hidden', [])

        if object_filter == {}:
            if uuid == "":
                self.log('Object with no filter/uuid requested:', schema,
                         data,
                         lvl=warn)
                return
            object_filter = {'uuid': uuid}

        storage_object = None
        storage_object = objectmodels[schema].find_one(object_filter)

        if not storage_object:
            self._cancel_by_error(event, uuid + '(' + str(object_filter) + ') of ' + schema +
                                  ' unavailable')
            return

        if storage_object:
            self.log("Object found, checking permissions: ", data, lvl=verbose)

            if not self._check_permissions(user, 'read',
                                           storage_object):
                self._cancel_by_permission(schema, data, event)
                return

            for field in hidden:
                storage_object._fields.pop(field, None)

            if do_subscribe and uuid != "":
                self._add_subscription(uuid, event)

            result = {
                'component': 'hfos.events.objectmanager',
                'action': 'get',
                'data': {
                    'schema': schema,
                    'uuid': uuid,
                    'object': storage_object.serializablefields()
                }
            }
            self._respond(None, result, event)
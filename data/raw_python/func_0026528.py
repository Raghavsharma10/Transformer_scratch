def put(self, event):
        """Put an object"""

        try:
            data, schema, user, client = self._get_args(event)
        except AttributeError:
            return

        try:
            clientobject = data['obj']
            uuid = clientobject['uuid']
        except KeyError as e:
            self.log("Put request with missing arguments!", e, data,
                     lvl=critical)
            return

        try:
            model = objectmodels[schema]
            created = False
            storage_object = None

            if uuid != 'create':
                storage_object = model.find_one({'uuid': uuid})
            if uuid == 'create' or model.count({'uuid': uuid}) == 0:
                if uuid == 'create':
                    uuid = str(uuid4())
                created = True
                clientobject['uuid'] = uuid
                clientobject['owner'] = user.uuid
                storage_object = model(clientobject)
                if not self._check_create_permission(user, schema):
                    self._cancel_by_permission(schema, data, event)
                    return

            if storage_object is not None:
                if not self._check_permissions(user, 'write', storage_object):
                    self._cancel_by_permission(schema, data, event)
                    return

                self.log("Updating object:", storage_object._fields, lvl=debug)
                storage_object.update(clientobject)

            else:
                storage_object = model(clientobject)
                if not self._check_permissions(user, 'write', storage_object):
                    self._cancel_by_permission(schema, data, event)
                    return

                self.log("Storing object:", storage_object._fields, lvl=debug)
                try:
                    storage_object.validate()
                except ValidationError:
                    self.log("Validation of new object failed!", clientobject,
                             lvl=warn)

            storage_object.save()

            self.log("Object %s stored." % schema)

            # Notify backend listeners

            if created:
                notification = objectcreation(
                    storage_object.uuid, schema, client
                )
            else:
                notification = objectchange(
                    storage_object.uuid, schema, client
                )

            self._update_subscribers(schema, storage_object)

            result = {
                'component': 'hfos.events.objectmanager',
                'action': 'put',
                'data': {
                    'schema': schema,
                    'object': storage_object.serializablefields(),
                    'uuid': storage_object.uuid,
                }
            }

            self._respond(notification, result, event)

        except Exception as e:
            self.log("Error during object storage:", e, type(e), data,
                     lvl=error, exc=True, pretty=True)
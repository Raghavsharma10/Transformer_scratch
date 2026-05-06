def delete(self, event):
        """Delete an existing object"""

        try:
            data, schema, user, client = self._get_args(event)
        except AttributeError:
            return

        try:
            uuids = data['uuid']

            if not isinstance(uuids, list):
                uuids = [uuids]

            if schema not in objectmodels.keys():
                self.log("Unknown schema encountered: ", schema, lvl=warn)
                return

            for uuid in uuids:
                self.log("Looking for object to be deleted:", uuid, lvl=debug)
                storage_object = objectmodels[schema].find_one({'uuid': uuid})

                if not storage_object:
                    self._cancel_by_error(event, 'not found')
                    return

                self.log("Found object.", lvl=debug)

                if not self._check_permissions(user, 'write', storage_object):
                    self._cancel_by_permission(schema, data, event)
                    return

                # self.log("Fields:", storage_object._fields, "\n\n\n",
                #         storage_object.__dict__)

                storage_object.delete()

                self.log("Deleted. Preparing notification.", lvl=debug)
                notification = objectdeletion(uuid, schema, client)

                if uuid in self.subscriptions:
                    deletion = {
                        'component': 'hfos.events.objectmanager',
                        'action': 'deletion',
                        'data': {
                            'schema': schema,
                            'uuid': uuid,
                        }
                    }
                    for recipient in self.subscriptions[uuid]:
                        self.fireEvent(send(recipient, deletion))

                    del (self.subscriptions[uuid])

                result = {
                    'component': 'hfos.events.objectmanager',
                    'action': 'delete',
                    'data': {
                        'schema': schema,
                        'uuid': storage_object.uuid
                    }
                }

                self._respond(notification, result, event)

        except Exception as e:
            self.log("Error during delete request: ", e, type(e),
                     lvl=error)
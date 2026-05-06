def _handleAuthorizedEvents(self, component, action, data, user, client):
        """Isolated communication link for authorized events."""

        try:
            if component == "debugger":
                self.log(component, action, data, user, client, lvl=info)

            if not user and component in self.authorized_events.keys():
                self.log("Unknown client tried to do an authenticated "
                         "operation: %s",
                         component, action, data, user)
                return

            event = self.authorized_events[component][action]['event'](user, action, data, client)

            self.log('Authorized event roles:', event.roles, lvl=verbose)
            if not self._checkPermissions(user, event):
                result = {
                    'component': 'hfos.ui.clientmanager',
                    'action': 'Permission',
                    'data': _('You have no role that allows this action.', lang='de')
                }
                self.fireEvent(send(event.client.uuid, result))
                return

            self.log("Firing authorized event: ", component, action,
                     str(data)[:100], lvl=debug)
            # self.log("", (user, action, data, client), lvl=critical)
            self.fireEvent(event)
        except Exception as e:
            self.log("Critical error during authorized event handling:",
                     component, action, e,
                     type(e), lvl=critical, exc=True)
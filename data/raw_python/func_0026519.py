def put(self, event):
        """Store a given configuration"""

        self.log("Configuration put request ",
                 event.user)

        try:
            component = model_factory(Schema).find_one({
                'uuid': event.data['uuid']
            })

            component.update(event.data)
            component.save()

            response = {
                'component': 'hfos.ui.configurator',
                'action': 'put',
                'data': True
            }
            self.log('Updated component configuration:',
                     component.name)

            self.fireEvent(reload_configuration(component.name))
        except (KeyError, ValueError, ValidationError, PermissionError) as e:
            response = {
                'component': 'hfos.ui.configurator',
                'action': 'put',
                'data': False
            }
            self.log('Storing component configuration failed: ',
                     type(e), e, exc=True, lvl=error)

        self.fireEvent(send(event.client.uuid, response))
        return
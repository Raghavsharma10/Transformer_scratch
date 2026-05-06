def get(self, event):
        """Get a stored configuration"""

        try:
            comp = event.data['uuid']
        except KeyError:
            comp = None

        if not comp:
            self.log('Invalid get request without schema or component',
                     lvl=error)
            return

        self.log("Config data get  request for ", event.data, "from",
                 event.user)

        component = model_factory(Schema).find_one({
            'uuid': comp
        })
        response = {
            'component': 'hfos.ui.configurator',
            'action': 'get',
            'data': component.serializablefields()
        }
        self.fireEvent(send(event.client.uuid, response))
def trigger(self, event):
        """AlertManager event handler for incoming events

        :param event with incoming AlertManager message
        """

        topic = event.data.get('topic', None)
        if topic is None:
            self.log('No alert topic to trigger', lvl=warn)
            return

        alert = {
            'topic': topic,
            'message': event.data.get('msg', 'Alert has been triggered'),
            'role': event.data.get('role', 'all')
        }

        self._trigger(event, alert)
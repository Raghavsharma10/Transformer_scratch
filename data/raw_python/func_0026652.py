def cancel(self, event):
        """AlertManager event handler for incoming events

        :param event with incoming AlertManager message
        """

        topic = event.data.get('topic', None)
        if topic is None:
            self.log('No alert topic to cancel', lvl=warn)
            return
        self._cancel(topic)
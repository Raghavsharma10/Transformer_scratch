def send_event(self, event_name, data, channel_name=None):
        """Send an event to the Pusher server.

        :param str event_name:
        :param Any data:
        :param str channel_name:
        """
        event = {'event': event_name, 'data': data}
        if channel_name:
            event['channel'] = channel_name

        self.logger.info("Connection: Sending event - %s" % event)
        try:
            self.socket.send(json.dumps(event))
        except Exception as e:
            self.logger.error("Failed send event: %s" % e)
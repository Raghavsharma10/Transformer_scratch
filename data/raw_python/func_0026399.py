def join(self, event):
        """Chat event handler for incoming events
        :param event: say-event with incoming chat message
        """

        try:
            channel_uuid = event.data
            user_uuid = event.user.uuid

            if channel_uuid in self.chat_channels:
                self.log('User joins a known channel', lvl=debug)
                if user_uuid in self.chat_channels[channel_uuid].users:
                    self.log('User already joined', lvl=warn)
                else:
                    self.chat_channels[channel_uuid].users.append(user_uuid)
                    self.chat_channels[channel_uuid].save()
                    packet = {
                        'component': 'hfos.chat.host',
                        'action': 'join',
                        'data': channel_uuid
                    }
                    self.fireEvent(send(event.client.uuid, packet))
            else:
                self.log('Request to join unavailable channel', lvl=warn)
        except Exception as e:
            self.log('Join error:', e, type(e), exc=True, lvl=error)
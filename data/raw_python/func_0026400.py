def say(self, event):
        """Chat event handler for incoming events
        :param event: say-event with incoming chat message
        """

        try:
            userid = event.user.uuid
            recipient = self._get_recipient(event)
            content = self._get_content(event)

            message = objectmodels['chatmessage']({
                'timestamp': time(),
                'recipient': recipient,
                'sender': userid,
                'content': content,
                'uuid': std_uuid()
            })

            message.save()

            chat_packet = {
                'component': 'hfos.chat.host',
                'action': 'say',
                'data': message.serializablefields()
            }

            if recipient in self.chat_channels:
                for useruuid in self.users:
                    if useruuid in self.chat_channels[recipient].users:
                        self.log('User in channel', lvl=debug)
                        self.update_lastlog(useruuid, recipient)

                        self.log('Sending message', lvl=debug)
                        self.fireEvent(send(useruuid, chat_packet,
                                            sendtype='user'))

        except Exception as e:
            self.log("Error: '%s' %s" % (e, type(e)), exc=True, lvl=error)
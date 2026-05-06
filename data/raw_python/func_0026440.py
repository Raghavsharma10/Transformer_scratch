def say(self, event):
        """Chat event handler for incoming events
        :param event: say-event with incoming chat message
        """

        try:
            userid = event.user.uuid
            recipient = self._get_recipient(event)
            content = self._get_content(event)

            if self.config.name in content:
                self.log('I think, someone mentioned me:', content)

        except Exception as e:
            self.log("Error: '%s' %s" % (e, type(e)), exc=True, lvl=error)
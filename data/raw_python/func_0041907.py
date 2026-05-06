def __on_message(self, msg):
        """
        XMPP message received
        """
        msgtype = msg['type']
        msgfrom = msg['from']
        if msgtype == 'groupchat':
            # MUC Room chat
            if self._nick == msgfrom.resource:
                # Loopback message
                return
        elif msgtype not in ('normal', 'chat'):
            # Ignore non-chat messages
            return

        # Callback
        self.__callback(msg)
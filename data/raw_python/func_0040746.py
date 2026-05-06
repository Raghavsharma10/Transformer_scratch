def on_privmsg(self, connection, event):
        """
        Got a message from a channel
        """
        sender = self.get_nick(event.source)
        message = event.arguments[0]
        
        if sender == 'NickServ':
            logging.info("Got message from NickServ: %s", message)
            if "password" in message.lower():
                connection.privmsg("NickServ", "pass")
            else:
                connection.join('#cohorte')
            
            return
        
        self._pool.enqueue(self.__on_message, connection, sender, message)
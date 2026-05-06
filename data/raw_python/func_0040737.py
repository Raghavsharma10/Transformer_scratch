def on_privmsg(self, connection, event):
        """
        Got a message from a user
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

        self.handle_message(connection, sender, sender, message)
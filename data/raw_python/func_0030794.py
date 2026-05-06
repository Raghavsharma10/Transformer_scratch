def messagesReceived(self, msg_list):
        """ Handle incoming messages

        @param msg_list: Message list to process
        """
        self.stats.packReceived(len(msg_list))

        for msg in msg_list:
            self.conn.messageReceived(msg)
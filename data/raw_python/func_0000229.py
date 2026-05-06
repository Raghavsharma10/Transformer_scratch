def sendQuery(self, cmd, multilines=False):
        """ Send command, wait for response (single or multi lines), test for errors and return the returned code.

        :param cmd: command to send
        :param multilines: True - multiline response, False - single line response.
        :return: command return value.
        """
        self.logger.debug("sendQuery(%s)", cmd)
        if not self.is_connected():
            raise socket.error("sendQuery on a disconnected socket")

        if multilines:
            replies = self.__sendQueryReplies(cmd)
            for reply in replies:
                if reply.startswith(XenaSocket.reply_errors):
                    raise XenaCommandException('sendQuery({}) reply({})'.format(cmd, replies))
            self.logger.debug("sendQuery(%s) -- Begin", cmd)
            for l in replies:
                self.logger.debug("%s", l.strip())
            self.logger.debug("sendQuery(%s) -- End", cmd)
            return replies
        else:
            reply = self.__sendQueryReply(cmd)
            if reply.startswith(XenaSocket.reply_errors):
                raise XenaCommandException('sendQuery({}) reply({})'.format(cmd, reply))
            self.logger.debug('sendQuery(%s) reply(%s)', cmd, reply)
            return reply
def sendQueryVerify(self, cmd):
        """ Send command without return value, wait for completion, verify success.

        :param cmd: command to send
        """
        cmd = cmd.strip()
        self.logger.debug("sendQueryVerify(%s)", cmd)
        if not self.is_connected():
            raise socket.error("sendQueryVerify on a disconnected socket")

        resp = self.__sendQueryReply(cmd)
        if resp != self.reply_ok:
            raise XenaCommandException('Command {} Fail Expected {} Actual {}'.format(cmd, self.reply_ok, resp))
        self.logger.debug("SendQueryVerify(%s) Succeed", cmd)
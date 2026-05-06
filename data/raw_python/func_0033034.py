def send_file(self, file):
        """
        Send a file to the client, it is a convenient method to avoid duplicated code
        """
        if self.logger:
            self.logger.debug("[ioc.extra.tornado.RouterHandler] send file %s" % file)

        self.send_file_header(file)

        fp = open(file, 'rb')
        self.write(fp.read())

        fp.close()
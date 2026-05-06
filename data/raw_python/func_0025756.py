def handle(self, data):
        """
        puts the data in the target.
        :param data: the data to post.
        :return:
        """
        self.dataResponseCode.append(self._doPut(self.sendURL + '/data', data=data))
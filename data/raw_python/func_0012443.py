def select_page(self, limit, offset=0, **kwargs):
        """
        :type limit: int
        :param limit: The max row number for each page
        :type offset: int
        :param offset: The starting position of the page
        :return:
        """
        start = offset
        while True:
            result = self.select(limit=[start, limit], **kwargs)
            start += limit
            if result:
                yield result
            else:
                break
            if self.debug:
                break
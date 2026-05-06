def chunk(self, count):
        """
        Chunk the results of the query

        :param count: The chunk size
        :type count: int

        :return: The current chunk
        :rtype: list
        """
        page = 1
        results = self.for_page(page, count).get()

        while results:
            yield results

            page += 1

            results = self.for_page(page, count).get()
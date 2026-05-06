def createFromFile(self, sid, filename):
        """
        Create a new named (sid) Seed from a file containing URLs
        It's assumed URLs are whitespace seperated.

        :param sid: the name to assign to the new seed list
        :param filename: the name of the file that contains URLs
        :return: the created Seed object
        """

        urls = []
        with open(filename) as f:
            for line in f:
                for url in line.split():
                    urls.append(url)

        return self.create(sid, tuple(urls))
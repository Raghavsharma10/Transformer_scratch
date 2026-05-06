def add_fileformat(self, fileformat):
        """
        Add fileformat line to the header.

        Arguments:
            fileformat (str): The id of the info line

        """
        self.fileformat = fileformat
        logger.info("Adding fileformat to vcf: {0}".format(fileformat))
        return
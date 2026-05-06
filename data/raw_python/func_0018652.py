def process_header(self, header):
        """
        Parse the header for additional metadata.

        Parameters
        ----------
        header : str
            All the lines in the header.

        Returns
        -------
        dict
            The metadata in the header.
        """
        metadata = {}
        for line in header.split("\n"):
            line = line.strip()
            for tag in self.header_tags:
                tag_text = "{}:".format(tag)
                if line.lower().startswith(tag_text):
                    metadata[tag] = line[len(tag_text) + 1 :].strip()

        return metadata
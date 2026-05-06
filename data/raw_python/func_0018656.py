def process_header(self, data):
        """
        Reads the first part of the file to get some essential metadata

        # Returns
        return (dict): the metadata in the header
        """
        metadata = {
            "datacolumns": data.read_chunk("I"),
            "firstyear": data.read_chunk("I"),
            "lastyear": data.read_chunk("I"),
            "annualsteps": data.read_chunk("I"),
        }
        if metadata["annualsteps"] != 1:
            raise InvalidTemporalResError(
                "{}: Only annual files can currently be processed".format(self.filepath)
            )

        return metadata
def _dialect(self, filepath):
        """returns detected dialect of filepath and sets self.has_header
        if not passed in __init__ kwargs
        Arguments:
            filepath (str): filepath of target csv file
        """
        with open(filepath, self.read_mode) as csvfile:
            sample = csvfile.read(1024)
            dialect = csv.Sniffer().sniff(sample)
            if self.has_header == None:
                # detect header if header not specified
                self.has_header = csv.Sniffer().has_header(sample)
            csvfile.seek(0)
        return dialect
def read_collection(self, filename):
        """
        Reads and returns a collection of stop words into a file.
        """
        with open(filename, 'rb') as fd:
            lines = fd.read().decode('utf-8-sig').splitlines()
        collection = list(filter(bool, [line.strip() for line in lines]))
        return collection
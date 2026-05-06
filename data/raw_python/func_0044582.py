def write_collection(self, filename, collection):
        """
        Writes a collection of stop words into a file.
        """
        collection = sorted(list(collection))
        with open(filename, 'wb+') as fd:
            fd.truncate()
            fd.write('\n'.join(collection).encode('utf-8'))
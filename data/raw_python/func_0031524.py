def saveAsJSON(self, fp, writeBytes=False):
        """
        Write the records out as JSON. The first JSON object saved contains
        information about the DIAMOND algorithm.

        @param fp: A C{str} file pointer to write to.
        @param writeBytes: If C{True}, the JSON will be written out as bytes
            (not strings). This is required when we are writing to a BZ2 file.
        """
        if writeBytes:
            fp.write(dumps(self.params, sort_keys=True).encode('UTF-8'))
            fp.write(b'\n')
            for record in self.records():
                fp.write(dumps(record, sort_keys=True).encode('UTF-8'))
                fp.write(b'\n')
        else:
            fp.write(six.u(dumps(self.params, sort_keys=True)))
            fp.write(six.u('\n'))
            for record in self.records():
                fp.write(six.u(dumps(record, sort_keys=True)))
                fp.write(six.u('\n'))
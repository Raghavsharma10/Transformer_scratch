def open_stream(self, class_attr_name=None, fn=None):
        """
        Save an arff structure to a file, leaving the file object
        open for writing of new data samples.
        This prevents you from directly accessing the data via Python,
        but when generating a huge file, this prevents all your data
        from being stored in memory.
        """
        if fn:
            self.fout_fn = fn
        else:
            fd, self.fout_fn = tempfile.mkstemp()
            os.close(fd)
        self.fout = open(self.fout_fn, 'w')
        if class_attr_name:
            self.class_attr_name = class_attr_name
        self.write(fout=self.fout, schema_only=True)
        self.write(fout=self.fout, data_only=True)
        self.fout.flush()
def from_file(self, fname):
        """read in a file and compute digest"""
        f = open(fname, "rb")
        data = f.read()
        self.update(data)
        f.close()
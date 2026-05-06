def dump(self, f, name):
        """Write the attribute to a file-like object"""
        # print the header line
        value = self.get()
        kind = self.get_kind(value)
        print("% 40s  kind=%s  value=%s" % (name, kind, value), file=f)
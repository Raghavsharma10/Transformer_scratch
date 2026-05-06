def open_file(self) :
        """If this FSNode is a file, open it for reading and return the file handle"""
        if self.isdir() : raise Exception("FSQuery tried to open a directory as a file : %s" % self.abs)
        return open(self.abs)
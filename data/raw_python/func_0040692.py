def add_file(self,fName,content) :
        """If this FSNode is a directory, write a file called fName containing content inside it"""
        if not self.isdir() : raise Exception("FSQuery tried to add a file in a node which is not a directory : %s" % self.abs)
        self.write_file("%s/%s"%(self.abs,fName),content)
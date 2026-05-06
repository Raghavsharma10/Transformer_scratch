def initread(self, idfname):
        """
        Use the current IDD and read an IDF from file. If the IDD has not yet
        been initialised then this is done first.

        Parameters
        ----------
        idf_name : str
            Path to an IDF file.

        """
        with open(idfname, 'r') as _:
            # raise nonexistent file error early if idfname doesn't exist
            pass
        iddfhandle = StringIO(iddcurrent.iddtxt)
        if self.getiddname() == None:
            self.setiddname(iddfhandle)
        self.idfname = idfname
        self.read()
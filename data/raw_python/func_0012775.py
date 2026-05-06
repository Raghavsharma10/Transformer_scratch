def initreadtxt(self, idftxt):
        """
        Use the current IDD and read an IDF from text data. If the IDD has not
        yet been initialised then this is done first.

        Parameters
        ----------
        idftxt : str
            Text representing an IDF file.

        """
        iddfhandle = StringIO(iddcurrent.iddtxt)
        if self.getiddname() == None:
            self.setiddname(iddfhandle)
        idfhandle = StringIO(idftxt)
        self.idfname = idfhandle
        self.read()
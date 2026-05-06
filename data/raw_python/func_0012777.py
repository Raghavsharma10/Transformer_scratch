def initnew(self, fname):
        """
        Use the current IDD and create a new empty IDF. If the IDD has not yet
        been initialised then this is done first.

        Parameters
        ----------
        fname : str, optional
            Path to an IDF. This does not need to be set at this point.

        """
        iddfhandle = StringIO(iddcurrent.iddtxt)
        if self.getiddname() == None:
            self.setiddname(iddfhandle)
        idfhandle = StringIO('')
        self.idfname = idfhandle
        self.read()
        if fname:
            self.idfname = fname
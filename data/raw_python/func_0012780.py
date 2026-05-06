def copyidfobject(self, idfobject):
        """Add an IDF object to the IDF.

        Parameters
        ----------
        idfobject : EpBunch object
            The IDF object to remove. This usually comes from another idf file,
            or it can be used to copy within this idf file.

        """
        return addthisbunch(self.idfobjects,
                     self.model,
                     self.idd_info,
                     idfobject, self)
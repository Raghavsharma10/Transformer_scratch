def read(self):
        """
        Read the IDF file and the IDD file. If the IDD file had already been
        read, it will not be read again.

        Read populates the following data structures:

        - idfobjects : list
        - model : list
        - idd_info : list
        - idd_index : dict

        """
        if self.getiddname() == None:
            errortxt = ("IDD file needed to read the idf file. "
                        "Set it using IDF.setiddname(iddfile)")
            raise IDDNotSetError(errortxt)
        readout = idfreader1(
            self.idfname, self.iddname, self,
            commdct=self.idd_info, block=self.block)
        (self.idfobjects, block, self.model,
            idd_info, idd_index, idd_version) = readout
        self.__class__.setidd(idd_info, idd_index, block, idd_version)
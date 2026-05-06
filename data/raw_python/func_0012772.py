def setiddname(cls, iddname, testing=False):
        """
        Set the path to the EnergyPlus IDD for the version of EnergyPlus which
        is to be used by eppy.

        Parameters
        ----------
        iddname : str
            Path to the IDD file.
        testing : bool
            Flag to use if running tests since we may want to ignore the
            `IDDAlreadySetError`.

        Raises
        ------
        IDDAlreadySetError

        """
        if cls.iddname == None:
            cls.iddname = iddname
            cls.idd_info = None
            cls.block = None
        elif cls.iddname == iddname:
            pass
        else:
            if testing == False:
                errortxt = "IDD file is set to: %s" % (cls.iddname,)
                raise IDDAlreadySetError(errortxt)
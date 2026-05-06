def user_return(self, frame, return_value):
        """This function is called when a return trap is set here."""
        pdb.Pdb.user_return(self, frame, return_value)
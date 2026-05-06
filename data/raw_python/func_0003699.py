def user_exception(self, frame, exc_info):
        """This function is called if an exception occurs,
        but only if we are to stop at or just below this level."""
        pdb.Pdb.user_exception(self, frame, exc_info)
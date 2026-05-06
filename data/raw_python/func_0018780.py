def remove_temp_copy(self):
        """
        Removes a temporary copy of the MAGICC version shipped with Pymagicc.
        """
        if self.is_temp and self.root_dir is not None:
            shutil.rmtree(self.root_dir)
            self.root_dir = None
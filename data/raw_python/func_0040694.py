def mk_dir(self) :
        """If this FSNode doesn't currently exist, then make a directory with this name."""
        if not os.path.exists(self.abs) :
            os.makedirs(self.abs)
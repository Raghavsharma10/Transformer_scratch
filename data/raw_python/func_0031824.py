def _set_up_savefolder(self):
        """ Create catalogs for different file output to clean up savefolder.
        """
        if not os.path.isdir(self.cells_path):
            os.mkdir(self.cells_path)

        if not os.path.isdir(self.figures_path):
            os.mkdir(self.figures_path)

        if not os.path.isdir(self.populations_path):
            os.mkdir(self.populations_path)
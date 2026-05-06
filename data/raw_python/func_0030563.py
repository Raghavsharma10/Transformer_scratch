def reset(self):
        """ Resets index by removing index directory. """
        if os.path.exists(self.index_dir):
            rmtree(self.index_dir)
        self.index = None
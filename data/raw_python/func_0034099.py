def rm(self, index):
        """Handles the 'r' command.

        :index: Index of the item to remove.

        """
        if self.model.exists(index):
            self.model.remove(index)
def undone(self, index):
        """Handles the 'D' command.

        :index: Index of the item to mark as not done.

        """
        if self.model.exists(index):
            self.model.edit(index, done=False)
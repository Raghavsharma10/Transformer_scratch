def done(self, index):
        """Handles the 'd' command.

        :index: Index of the item to mark as done.

        """
        if self.model.exists(index):
            self.model.edit(index, done=True)
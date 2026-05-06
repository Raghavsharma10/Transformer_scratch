def poll_error(self):
        """
        Append lines from stderr to self.errors.

        Returns:
            list: The lines added since last call
        """
        if self.block:
            return self.error

        new_list = self.error[self.old_error_size:]
        self.old_error_size += len(new_list)
        return new_list
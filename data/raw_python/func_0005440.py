def poll_output(self):
        """
        Append lines from stdout to self.output.

        Returns:
            list: The lines added since last call
        """
        if self.block:
            return self.output

        new_list = self.output[self.old_output_size:]
        self.old_output_size += len(new_list)
        return new_list
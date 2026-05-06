def unhandled_keys(self, size, key):
        """
        Override this method to intercept keystrokes in subclasses.
        Default behavior: Toggle flagged on space, ignore other keys.
        """
        if key == " ":
            if not self.flagged:
                self.display.new_files.append(self.get_node().get_value())
            else:
                self.display.new_files.remove(self.get_node().get_value())
            self.flagged = not self.flagged
            self.update_w()
            self.display.update_status()
        else:
            return key
def disk2ram(self):
        """Move internal data from disk to RAM."""
        values = self.series
        self.deactivate_disk()
        self.ramflag = True
        self.__set_array(values)
        self.update_fastaccess()
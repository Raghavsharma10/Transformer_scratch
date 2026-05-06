def update(self, progress=0):
        """ Updates the progress bar with @progress if given, otherwise
            increments :prop:progress by 1. Also prints the progress bar.

            @progress: #int to assign to :prop:progress
        """
        self.progress += (progress or 1)
        if self.visible:
            if self.progress % self._mod == 1 or\
               self.progress == self.size - 1:
                print(self.format_bar(), end="\r")
            if self.progress == (self.size):
                self.finish()
def iterate(self, iterable, format="%s"):
        """Use as a target of a for-loop to issue a progress update for every
        iteration. For example:

        progress = ProgressBar()
        for text in progress.iterate(["foo", "bar", "bat"]):
            ...
        """

        # If iterable has a definite length, then set the maximum value of the
        # progress bar. Else, set the maximum value to -1 so that the progress
        # bar displays indeterminate progress (scrolling dots).
        try:
            length = len(iterable)
        except TypeError:
            self.max = -1
        else:
            self.max = length

        # Iterate over the input, updating the progress bar for each element.
        for i, item in enumerate(iterable):
            self.update(i, format % item)
            yield item
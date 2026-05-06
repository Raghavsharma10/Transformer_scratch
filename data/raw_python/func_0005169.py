def format_bar(self):
        """ Builds the progress bar """
        pct = floor(round(self.progress/self.size, 2)*100)
        pr = floor(pct*.33)
        bar = "".join(
            ["‒" for x in range(pr)] + ["↦"] +
            [" " for o in range(self._barsize-pr-1)])
        subprogress = self.format_parent_bar() if self.parent_bar else ""
        message = "Loading{} ={}{} ({}%)".format(subprogress, bar, "☉", pct)
        return message.ljust(len(message)+5)
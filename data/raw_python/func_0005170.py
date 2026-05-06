def finish(self):
        """ Resets the progress bar and clears it from the terminal """
        pct = floor(round(self.progress/self.size, 2)*100)
        pr = floor(pct*.33)
        bar = "".join([" " for x in range(pr-1)] + ["↦"])
        subprogress = self.format_parent_bar() if self.parent_bar else ""
        fin = "Loading{} ={}{} ({}%)".format(subprogress, bar, "ӿ", pct)
        print(fin.ljust(len(fin)+5), end="\r")
        time.sleep(0.10)
        print("\033[K\033[1A")
        self.progress = 0
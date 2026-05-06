def save(self, filename=None, debug=False):
        """save a data file such that all processes know the game that is running"""
        if not filename: filename = self.name
        with open(filename, "w") as f: # save config data file
            f.write(self.toJson(self.attrs))
        if self.debug or debug:
            print("saved configuration %s"%(self.name))
            for k,v in sorted(iteritems(self.attrs)):
                print("%15s : %s"%(k,v))
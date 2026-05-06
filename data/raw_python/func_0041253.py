def load(self, ladderName):
        """retrieve the ladder settings from saved disk file"""
        self.name = ladderName # preset value to load self.filename
        with open(self.filename, "rb") as f:
            data = f.read()
            self.__dict__.update( json.loads(data) )
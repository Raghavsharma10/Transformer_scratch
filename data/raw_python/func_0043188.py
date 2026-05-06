def load(self):
        """load ALL_VERS_DATA from disk"""
        basepath = os.path.dirname(os.path.abspath(__file__))
        filename = os.sep.join([basepath, c.FOLDER_JSON, c.FILE_GAME_VERSIONS])
        Handler.ALL_VERS_DATA = {} # reset known data; do not retain defunct information
        with open(filename, "r") as f:
            data = json.loads( f.read() )
        self.update(data)
        self._updated = False
def do_mkdir(self, path):
        """create a new directory"""
        path = path[0]

        self.n.makeDirectory(self.current_path + path)
        self.dirs = self.dir_complete()
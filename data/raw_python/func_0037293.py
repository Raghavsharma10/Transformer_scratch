def do_rm(self, path):
        path = path[0]

        """delete a file or directory"""
        self.n.delete(self.current_path + path)
        self.dirs = self.dir_complete()
        self.files = self.file_complete()
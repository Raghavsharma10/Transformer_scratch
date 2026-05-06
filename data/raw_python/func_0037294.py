def do_mv(self, from_path, to_path, nothing = ''):
        """move/rename a file or directory"""
        self.n.doMove(self.current_path + from_path,
                      self.current_path + to_path)
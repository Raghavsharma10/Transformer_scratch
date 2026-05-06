def move_file(self, src, dst):
        """ Move file from src to dst """

        src = self.get_full_file_path(src); dst = self.get_full_file_path(dst)

        # record where file moved
        if os.path.isfile(src):
            # if destination file exists, copy it to tmp first
            if os.path.isfile(dst):
                tmp_path = self.new_tmp()
                self.do_action({
                    'do'   : ['copy', dst, tmp_path],
                    'undo' : ['move', tmp_path, dst]})

        self.do_action(
            {'do'   : ['move', src, dst],
             'undo' : ['move', dst, src]})
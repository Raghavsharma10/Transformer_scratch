def file_put_contents(self, path, data):
        """ Put passed contents into file located at 'path' """

        path = self.get_full_file_path(path)

        # if file exists, create a temp copy to allow rollback
        if os.path.isfile(path):
            tmp_path = self.new_tmp()
            self.do_action({
                'do'   : ['copy', path, tmp_path],
                'undo' : ['move', tmp_path, path]})

        self.do_action(
            {'do'   : ['write', path, data],
             'undo' : ['backup', path]})
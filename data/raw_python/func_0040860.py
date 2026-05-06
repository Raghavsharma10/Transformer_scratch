def delete_file(self, path):
        """ delete a file """

        path = self.get_full_file_path(path)

        # if file exists, create a temp copy to allow rollback
        if os.path.isfile(path):
            tmp_path = self.new_tmp()
            self.do_action({
                'do'   : ['move', path, tmp_path],
                'undo' : ['move', tmp_path, path]})

        else:
            raise OSError(errno.ENOENT, 'No such file or directory', path)
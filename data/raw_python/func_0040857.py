def file_get_contents(self, path):
        """ Returns contents of file located at 'path', not changing FS so does
        not require journaling """

        with open(self.get_full_file_path(path), 'r') as f: return  f.read()
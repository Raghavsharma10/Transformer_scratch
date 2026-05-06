def os_path_transform(self, s, sep=os.path.sep):
        """ transforms any os path into unix style """
        if sep == '/':
            return s
        else:
            return s.replace(sep, '/')
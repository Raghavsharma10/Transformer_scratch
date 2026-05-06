def remove_file(filepath):
        ''' Delete a file '''
        try:
            os.remove(os.path.abspath(os.path.expanduser(filepath)))
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
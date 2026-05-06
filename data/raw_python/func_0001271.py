def replace_ext(file_path, ext):
        ''' Change extension of a file_path to something else (provide None to remove) '''
        if not file_path:
            raise Exception("File path cannot be empty")
        dirname = os.path.dirname(file_path)
        filename = FileHelper.getfilename(file_path)
        if ext:
            filename = filename + '.' + ext
        return os.path.join(dirname, filename)
def replace_name(file_path, new_name):
        ''' Change the file name in a path but keep the extension '''
        if not file_path:
            raise Exception("File path cannot be empty")
        elif not new_name:
            raise Exception("New name cannot be empty")
        dirname = os.path.dirname(file_path)
        ext = os.path.splitext(os.path.basename(file_path))[1]
        return os.path.join(dirname, new_name + ext)
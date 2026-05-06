def get_child_files(path):
        ''' Get all child files of a folder '''
        path = FileHelper.abspath(path)
        return [filename for filename in os.listdir(path) if os.path.isfile(os.path.join(path, filename))]
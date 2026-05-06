def get_child_folders(path):
        ''' Get all child folders of a folder '''
        path = FileHelper.abspath(path)
        return [dirname for dirname in os.listdir(path) if os.path.isdir(os.path.join(path, dirname))]
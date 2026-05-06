def force_delete_file(file_path):
        ''' force delete a file '''
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                return file_path
            except:           
                return FileSystemUtils.add_unique_postfix(file_path)
        else:
            return file_path
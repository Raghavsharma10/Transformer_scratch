def unzip(self, directory):
        """
        Write contents of zipfile to directory
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        shutil.copytree(self.src_dir, directory)
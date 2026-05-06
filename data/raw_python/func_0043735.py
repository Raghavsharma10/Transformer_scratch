def settings_dir(self):
        """
        Directory that contains the the settings for the project
        """
        path = os.path.join(self.dir, '.dsb')
        utils.create_dir(path)
        return os.path.realpath(path)
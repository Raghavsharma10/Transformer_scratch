def load(self, filepath=None):
        """
        Load settings file from given path and optionnal filepath.

        During path resolving, the ``projectdir`` is updated to the file path
        directory.

        Keyword Arguments:
            filepath (str): Filepath to the settings file.

        Returns:
            boussole.conf.model.Settings: Settings object with loaded options.

        """
        self.projectdir, filename = self.parse_filepath(filepath)

        settings_path = self.check_filepath(self.projectdir, filename)

        parsed = self.parse(settings_path, self.open(settings_path))

        settings = self.clean(parsed)

        return Settings(initial=settings)
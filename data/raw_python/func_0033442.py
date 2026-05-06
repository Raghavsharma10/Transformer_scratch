def _config_bootstrap(self) -> None:
        """Handle the basic setup of the tool prior to user control.

        Bootstrap will load all the available modules for searching and set
        them up for use by this main class.
        """
        if self.output:
            self.folder: str = os.getcwd() + "/" + self.project
            os.mkdir(self.folder)
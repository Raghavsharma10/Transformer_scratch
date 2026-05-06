def add_config(self, config):
        """
        :param config:
        :type config: dict
        """
        self.pre_configure()

        self.config = config

        if not self.has_revision_file():
            #: Create new revision file.
            touch_file(self.revfile_path)

        self.history.load(self.revfile_path)

        self.archiver.target_path = self.dest_path
        self.archiver.zip_path = self.tmp_file_path

        self.state.state_path = os.path.join(
            REVISION_HOME,
            "clients",
            self.key
        )
        self.state.prepare()

        self.post_configure()

        self.prepared = True
def create_copy(self):
        """
        Initialises a temporary directory structure and copy of MAGICC
        configuration files and binary.
        """
        if self.executable is None or not isfile(self.executable):
            raise FileNotFoundError(
                "Could not find MAGICC{} executable: {}".format(
                    self.version, self.executable
                )
            )
        if self.is_temp:
            assert (
                self.root_dir is None
            ), "A temp copy for this instance has already been created"
            self.root_dir = mkdtemp(prefix="pymagicc-")

        if exists(self.run_dir):
            raise Exception("A copy of MAGICC has already been created.")
        if not exists(self.root_dir):
            makedirs(self.root_dir)

        exec_dir = basename(self.original_dir)

        # Copy a subset of folders from the MAGICC `original_dir`
        # Also copy anything which is in the root of the MAGICC distribution
        # Assumes that the MAGICC binary is in a folder one level below the root
        # of the MAGICC distribution. i.e. /run/magicc.exe or /bin/magicc
        dirs_to_copy = [".", "bin", "run"]
        # Check that the executable is in a valid sub directory
        assert exec_dir in dirs_to_copy, "binary must be in bin/ or run/ directory"

        for d in dirs_to_copy:
            source_dir = abspath(join(self.original_dir, "..", d))
            if exists(source_dir):
                _copy_files(source_dir, join(self.root_dir, d))

        # Create an empty out dir
        # MAGICC assumes that the 'out' directory already exists
        makedirs(join(self.root_dir, "out"))

        # Create basic configuration files so magicc can run
        self.set_years()
        self.set_config()
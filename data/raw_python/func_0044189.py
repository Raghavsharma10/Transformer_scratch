def commit(self, sourcedir, targetdir, abs_config, abs_sourcedir,
               abs_targetdir):
        """
        Commit project structure and configuration file

        Args:
            sourcedir (string): Source directory path.
            targetdir (string): Compiled files target directory path.
            abs_config (string): Configuration file absolute path.
            abs_sourcedir (string): ``sourcedir`` expanded as absolute path.
            abs_targetdir (string): ``targetdir`` expanded as absolute path.
        """
        config_path, config_filename = os.path.split(abs_config)

        if not os.path.exists(config_path):
            os.makedirs(config_path)
        if not os.path.exists(abs_sourcedir):
            os.makedirs(abs_sourcedir)
        if not os.path.exists(abs_targetdir):
            os.makedirs(abs_targetdir)

        # Dump settings file
        self.backend_engine.dump({
            'SOURCES_PATH': sourcedir,
            'TARGET_PATH': targetdir,
            "LIBRARY_PATHS": [],
            "OUTPUT_STYLES": "nested",
            "SOURCE_COMMENTS": False,
            "EXCLUDES": []
        }, abs_config, indent=4)
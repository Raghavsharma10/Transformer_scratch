def _readall(self):
        """Read configs from all available configs. It will read files in the following order:

            1.) Read all default settings:

                These are located under: `<project_root>/config/*/default.cfg`

            2.) Read the user's config settings:

                This is located on the path: `~/.aftrc`

            3.) Read all config files specified by the config string in the environment variable TEST_RUN_SETTING_CONFIG

                A config string such as "browser.headless,scripts.no_ssh" will read paths:

                    `<project_root>/config/browser/headless.cfg`
                    `<project_root>/config/scripts/no_ssh.cfg`

                OR a config string such as "<project_root>/config/browser/headless.cfg" will load that path directly
        """
        # First priority -- read all default configs
        config_path = os.path.dirname(__file__)
        config_defaults = [os.path.join(dirpath, f)
                           for dirpath, dirnames, files in os.walk(config_path)
                           for f in fnmatch.filter(files, 'default.cfg')]

        # Second priority -- read the user overrides
        user_config = os.path.expanduser('~/.aftrc')

        # Third priority -- read the environment variable overrides
        override_filenames = []
        if TEST_RUN_SETTING_CONFIG in os.environ:
            for test_config in os.environ[TEST_RUN_SETTING_CONFIG].split(','):
                if os.path.exists(test_config):             #is this a file path
                   override_filenames.append(test_config)
                elif "." in test_config and not test_config.endswith('.cfg'):                    #else it might be in xxxx.yyyy format
                    config_parts = test_config.split('.')
                    config_parts[-1]+='.cfg' #add file ext to last part, which should be file
                    filename = os.path.join(config_path, *config_parts)
                    override_filenames.append(filename)
                else:                                       #else unknown, might throw exception here
                    pass


        all_configs = config_defaults + [user_config] + override_filenames
        return self.parser.read(all_configs)
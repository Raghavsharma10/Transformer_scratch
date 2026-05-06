def get_config(self):
        """
        Reads the settings from the gce.ini file.

        Populates a SafeConfigParser object with defaults and
        attempts to read an .ini-style configuration from the filename
        specified in GCE_INI_PATH. If the environment variable is
        not present, the filename defaults to gce.ini in the current
        working directory.
        """
        gce_ini_default_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "gce.ini")
        gce_ini_path = os.environ.get('GCE_INI_PATH', gce_ini_default_path)

        # Create a ConfigParser.
        # This provides empty defaults to each key, so that environment
        # variable configuration (as opposed to INI configuration) is able
        # to work.
        config = ConfigParser.SafeConfigParser(defaults={
            'gce_service_account_email_address': '',
            'gce_service_account_pem_file_path': '',
            'gce_project_id': '',
            'libcloud_secrets': '',
            'inventory_ip_type': '',
            'cache_path': '~/.ansible/tmp',
            'cache_max_age': '300'
        })
        if 'gce' not in config.sections():
            config.add_section('gce')
        if 'inventory' not in config.sections():
            config.add_section('inventory')
        if 'cache' not in config.sections():
            config.add_section('cache')

        config.read(gce_ini_path)

        #########
        # Section added for processing ini settings
        #########

        # Set the instance_states filter based on config file options
        self.instance_states = []
        if config.has_option('gce', 'instance_states'):
            states = config.get('gce', 'instance_states')
            # Ignore if instance_states is an empty string.
            if states:
                self.instance_states = states.split(',')

        # Caching
        cache_path = config.get('cache', 'cache_path')
        cache_max_age = config.getint('cache', 'cache_max_age')
        # TOOD(supertom): support project-specific caches
        cache_name = 'ansible-gce.cache'
        self.cache = CloudInventoryCache(cache_path=cache_path,
                                         cache_max_age=cache_max_age,
                                         cache_name=cache_name)
        return config
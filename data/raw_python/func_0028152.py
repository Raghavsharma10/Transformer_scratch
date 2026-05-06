def _create(cls, configuration=None, remoteckan=None, **kwargs):
        # type: (Optional['Configuration'], Optional[ckanapi.RemoteCKAN], Any) -> str
        """
        Create HDX configuration

        Args:
            configuration (Optional[Configuration]): Configuration instance. Defaults to setting one up from passed arguments.
            remoteckan (Optional[ckanapi.RemoteCKAN]): CKAN instance. Defaults to setting one up from configuration.
            **kwargs: See below
            user_agent (str): User agent string. HDXPythonLibrary/X.X.X- is prefixed. Must be supplied if remoteckan is not.
            user_agent_config_yaml (str): Path to YAML user agent configuration. Ignored if user_agent supplied. Defaults to ~/.useragent.yml.
            user_agent_lookup (str): Lookup key for YAML. Ignored if user_agent supplied.
            hdx_url (str): HDX url to use. Overrides hdx_site.
            hdx_site (str): HDX site to use eg. prod, test.
            hdx_read_only (bool): Whether to access HDX in read only mode. Defaults to False.
            hdx_key (str): Your HDX key. Ignored if hdx_read_only = True.
            hdx_config_dict (dict): HDX configuration dictionary to use instead of above 3 parameters OR
            hdx_config_json (str): Path to JSON HDX configuration OR
            hdx_config_yaml (str): Path to YAML HDX configuration
            project_config_dict (dict): Project configuration dictionary OR
            project_config_json (str): Path to JSON Project configuration OR
            project_config_yaml (str): Path to YAML Project configuration
            hdx_base_config_dict (dict): HDX base configuration dictionary OR
            hdx_base_config_json (str): Path to JSON HDX base configuration OR
            hdx_base_config_yaml (str): Path to YAML HDX base configuration. Defaults to library's internal hdx_base_configuration.yml.

        Returns:
            str: HDX site url

        """
        kwargs = cls._environment_variables(**kwargs)
        cls.setup(configuration, **kwargs)
        cls._configuration.setup_remoteckan(remoteckan, **kwargs)
        return cls._configuration.get_hdx_site_url()
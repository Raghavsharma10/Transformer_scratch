def _create_configs(cls, site):
        """
        This is going to generate the following configuration:
        * wsgi.py
        * <provider>.yml
        * settings_<provider>.py
        """
        provider = cls.name

        cls._render_config('wsgi.py', 'wsgi.py', site)

        # create yaml file
        yaml_template_name = os.path.join(provider, cls.provider_yml_name)
        cls._render_config(cls.provider_yml_name, yaml_template_name, site)

        # create requirements file
        # don't do anything if the requirements file is called requirements.txt and in the root of the project
        requirements_filename = "requirements.txt"
        if site['requirements'] != requirements_filename:   # providers expect the file to be called requirements.txt
            requirements_template_name = os.path.join(provider, requirements_filename)
            cls._render_config(requirements_filename, requirements_template_name, site)

        # create settings file
        settings_template_name = os.path.join(provider, 'settings_%s.py' % provider)
        settings_path = site['django_settings'].replace('.', '/') + '_%s.py' % provider
        cls._render_config(settings_path, settings_template_name, site)
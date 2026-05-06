def load(cls, environment_name=None, site_name='primary', data_only=False, allow_old=False):
        """
        Return an Environment object based on an existing environnment+site.

        :param environment_name: exising environment name, path or None to
            look in current or parent directories for project
        :param data_only: set to True to only load from data dir, not
            the project dir; Used for purging environment data.
        :param allow_old: load a very minimal subset of what we usually
            load. This will only work for purging environment data on an old site.

        Raises DatacatsError if environment can't be found or if there is an
        error parsing the environment information.
        """
        srcdir, extension_dir, datadir = task.find_environment_dirs(
            environment_name, data_only)

        if datadir and data_only:
            return cls(environment_name, None, datadir, site_name)

        (datadir, name, ckan_version, always_prod, deploy_target,
            remote_server_key, extra_containers) = task.load_environment(srcdir, datadir, allow_old)

        if not allow_old:
            (port, address, site_url, passwords) = task.load_site(srcdir, datadir, site_name)
        else:
            (port, address, site_url, passwords) = (None, None, None, None)

        environment = cls(name, srcdir, datadir, site_name, ckan_version=ckan_version,
                          port=port, deploy_target=deploy_target, site_url=site_url,
                          always_prod=always_prod, address=address,
                          extension_dir=extension_dir,
                          remote_server_key=remote_server_key,
                          extra_containers=extra_containers)

        if passwords:
            environment.passwords = passwords
        else:
            environment._generate_passwords()

        if not allow_old:
            environment._load_sites()
        return environment
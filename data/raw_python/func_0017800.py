def new(cls, path, ckan_version, site_name, **kwargs):
        """
        Return a Environment object with settings for a new project.
        No directories or containers are created by this call.

        :params path: location for new project directory, may be relative
        :params ckan_version: release of CKAN to install
        :params site_name: The name of the site to install database and solr \
                            eventually.

        For additional keyword arguments see the __init__ method.

        Raises DatcatsError if directories or project with same
        name already exits.
        """
        if ckan_version == 'master':
            ckan_version = 'latest'
        name, datadir, srcdir = task.new_environment_check(path, site_name, ckan_version)
        environment = cls(name, srcdir, datadir, site_name, ckan_version, **kwargs)
        environment._generate_passwords()
        return environment
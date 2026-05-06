def write_config_from_api(self, api, config_file=None, profile=None):
        '''
        Create/update the config file from a DataAPI object

        Parameters
        ----------

        api : object
            The :py:class:`datafs.DataAPI` object from which
            to create the config profile

        profile : str
            Name of the profile to use in the config file
            (default "default-profile")

        config_file : str or file
            Path or file in which to write config (default
            is your OS's default datafs application
            directory)

        Examples
        --------

        Create a simple API and then write the config to a
        buffer:

        .. code-block:: python

            >>> from datafs import DataAPI
            >>> from datafs.managers.manager_mongo import MongoDBManager
            >>> from fs.osfs import OSFS
            >>> from fs.tempfs import TempFS
            >>> import os
            >>> import tempfile
            >>> import shutil
            >>>
            >>> api = DataAPI(
            ...      username='My Name',
            ...      contact = 'me@demo.com')
            >>>
            >>> manager = MongoDBManager(
            ...     database_name = 'MyDatabase',
            ...     table_name = 'DataFiles')
            >>>
            >>> manager.create_archive_table(
            ...     'DataFiles',
            ...     raise_on_err=False)
            >>>
            >>> api.attach_manager(manager)
            >>>
            >>> tmpdir = tempfile.mkdtemp()
            >>> local = OSFS(tmpdir)
            >>>
            >>> api.attach_authority('local', local)
            >>>
            >>> # Create a StringIO object for the config file
            ...
            >>> try:
            ...   from StringIO import StringIO
            ... except ImportError:
            ...   from io import StringIO
            ...
            >>> conf = StringIO()
            >>>
            >>> config_file = ConfigFile(default_profile='my-api')
            >>> config_file.write_config_from_api(
            ...     api,
            ...     profile='my-api',
            ...     config_file=conf)
            >>>
            >>> print(conf.getvalue())   # doctest: +SKIP
            default-profile: my-api
            profiles:
              my-api:
                api:
                  user_config: {contact: me@demo.com, username: My Name}
                authorities:
                  local:
                    args: [...]
                    service: OSFS
                    kwargs: {}
                manager:
                  args: []
                  class: MongoDBManager
                  kwargs:
                    client_kwargs: {}
                    database_name: MyDatabase
                    table_name: DataFiles
            <BLANKLINE>
            >>> conf.close()
            >>> local.close()
            >>> shutil.rmtree(tmpdir)

        At this point, we can retrieve the api object from
        the configuration file:

        .. code-block:: python

            >>> try:
            ...   from StringIO import StringIO
            ... except ImportError:
            ...   from io import StringIO
            ...
            >>> conf = StringIO("""
            ... default-profile: my-api
            ... profiles:
            ...   my-api:
            ...     api:
            ...       user_config: {contact: me@demo.com, username: My Name}
            ...     authorities:
            ...       local:
            ...         args: []
            ...         service: TempFS
            ...         kwargs: {}
            ...     manager:
            ...       args: []
            ...       class: MongoDBManager
            ...       kwargs:
            ...         client_kwargs: {}
            ...         database_name: MyDatabase
            ...         table_name: DataFiles
            ... """)
            >>>
            >>> import datafs
            >>> from fs.tempfs import TempFS
            >>> api = datafs.get_api(profile='my-api', config_file=conf)
            >>>
            >>> cache = TempFS()
            >>> api.attach_cache(cache)
            >>>
            >>> conf2 = StringIO()
            >>>
            >>> config_file = ConfigFile(default_profile='my-api')
            >>> config_file.write_config_from_api(
            ...     api,
            ...     profile='my-api',
            ...     config_file=conf2)
            >>>
            >>> print(conf2.getvalue())   # doctest: +SKIP
            default-profile: my-api
            profiles:
              my-api:
                api:
                  user_config: {contact: me@demo.com, username: My Name}
                authorities:
                  local:
                    args: []
                    service: TempFS
                    kwargs: {}
                cache:
                    args: []
                    service: TempFS
                    kwargs: {}
                manager:
                  args: []
                  class: MongoDBManager
                  kwargs:
                    client_kwargs: {}
                    database_name: MyDatabase
                    table_name: DataFiles
            <BLANKLINE>
        '''

        if profile is None:
            profile = self.default_profile

        self.get_config_from_api(api, profile)
        self.write_config(config_file)
def spawn_uwsgi(self, only=None):
        """Spawns uWSGI process(es) which will use configuration(s) from the module.

        Returns list of tuples:
            (configuration_alias, uwsgi_process_id)

        If only one configuration found current process (uwsgiconf) is replaced with a new one (uWSGI),
        otherwise a number of new detached processes is spawned.

        :param str|unicode only: Configuration alias to run from the module.
            If not set uWSGI will be spawned for every configuration found in the module.

        :rtype: list
        """
        spawned = []
        configs = self.configurations

        if len(configs) == 1:

            alias = configs[0].alias
            UwsgiRunner().spawn(self.fpath, alias, replace=True)
            spawned.append((alias, os.getpid()))

        else:
            for config in configs:  # type: Configuration
                alias = config.alias

                if only is None or alias == only:
                    pid = UwsgiRunner().spawn(self.fpath, alias)
                    spawned.append((alias, pid))

        return spawned
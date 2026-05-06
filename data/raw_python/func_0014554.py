def _expand_variables(self, config, hostname ):
        """
        Return a dict of config options with expanded substitutions
        for a given hostname.

        Please refer to man ssh_config(5) for the parameters that
        are replaced.

        @param config: the config for the hostname
        @type hostname: dict
        @param hostname: the hostname that the config belongs to
        @type hostname: str
        """

        if 'hostname' in config:
            config['hostname'] = config['hostname'].replace('%h',hostname)
        else:
            config['hostname'] = hostname

        if 'port' in config:
            port = config['port']
        else:
            port = SSH_PORT

        user = os.getenv('USER')
        if 'user' in config:
            remoteuser = config['user']
        else:
            remoteuser = user

        host = socket.gethostname().split('.')[0]
        fqdn = socket.getfqdn()
        homedir = os.path.expanduser('~')
        replacements = {'controlpath' :
                [
                    ('%h', config['hostname']),
                    ('%l', fqdn),
                    ('%L', host),
                    ('%n', hostname),
                    ('%p', port),
                    ('%r', remoteuser),
                    ('%u', user)
                ],
                'identityfile' :
                [
                    ('~', homedir),
                    ('%d', homedir),
                    ('%h', config['hostname']),
                    ('%l', fqdn),
                    ('%u', user),
                    ('%r', remoteuser)
                ]
                }
        for k in config:
            if k in replacements:
                for find, replace in replacements[k]:
                        config[k] = config[k].replace(find, str(replace))
        return config
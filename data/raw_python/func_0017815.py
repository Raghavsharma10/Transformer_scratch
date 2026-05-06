def _run_web_container(self, port, command, address, log_syslog=False,
                           datapusher=True, interactive=False):
        """
        Start web container on port with command
        """
        if is_boot2docker():
            ro = {}
            volumes_from = self._get_container_name('venv')
        else:
            ro = {self.datadir + '/venv': '/usr/lib/ckan'}
            volumes_from = None

        links = {
            self._get_container_name('solr'): 'solr',
            self._get_container_name('postgres'): 'db'
        }

        links.update({self._get_container_name(container): container
                      for container in self.extra_containers})

        if datapusher:
            if 'datapusher' not in self.containers_running():
                raise DatacatsError(container_logs(self._get_container_name('datapusher'), "all",
                                                   False, False))
            links[self._get_container_name('datapusher')] = 'datapusher'

        ro = dict({
                  self.target: '/project/',
                  scripts.get_script_path('web.sh'): '/scripts/web.sh',
                  scripts.get_script_path('adjust_devini.py'): '/scripts/adjust_devini.py'},
                  **ro)
        rw = {
            self.sitedir + '/files': '/var/www/storage',
            self.sitedir + '/run/development.ini': '/project/development.ini'
            }
        try:
            if not interactive:
                run_container(
                    name=self._get_container_name('web'),
                    image='datacats/web',
                    rw=rw,
                    ro=ro,
                    links=links,
                    volumes_from=volumes_from,
                    command=command,
                    port_bindings={
                        5000: port if is_boot2docker() else (address, port)},
                    log_syslog=log_syslog
                    )
            else:
                # FIXME: share more code with interactive_shell
                if is_boot2docker():
                    switches = ['--volumes-from',
                                self._get_container_name('pgdata'), '--volumes-from',
                                self._get_container_name('venv')]
                else:
                    switches = []
                switches += ['--volume={}:{}:ro'.format(vol, ro[vol]) for vol in ro]
                switches += ['--volume={}:{}'.format(vol, rw[vol]) for vol in rw]
                links = ['--link={}:{}'.format(link, links[link]) for link in links]
                args = ['docker', 'run', '-it', '--name', self._get_container_name('web'),
                        '-p', '{}:5000'.format(port) if is_boot2docker()
                        else '{}:{}:5000'.format(address, port)] + \
                    switches + links + ['datacats/web', ] + command
                subprocess.call(args)
        except APIError as e:
            if '409' in str(e):
                raise DatacatsError('Web container already running. '
                                    'Please stop_web before running.')
            else:
                raise
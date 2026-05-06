def start_ckan(self, production=False, log_syslog=False, paster_reload=True,
                   interactive=False):
        """
        Start the apache server or paster serve

        :param log_syslog: A flag to redirect all container logs to host's syslog
        :param production: True for apache, False for paster serve + debug on
        :param paster_reload: Instruct paster to watch for file changes
        """
        self.stop_ckan()

        address = self.address or '127.0.0.1'
        port = self.port
        # in prod we always use log_syslog driver
        log_syslog = True if self.always_prod else log_syslog

        production = production or self.always_prod
        # We only override the site URL with the docker URL on three conditions
        override_site_url = (self.address is None
                             and not is_boot2docker()
                             and not self.site_url)
        command = ['/scripts/web.sh', str(production), str(override_site_url), str(paster_reload)]

        # XXX nasty hack, remove this once we have a lessc command
        # for users (not just for building our preload image)
        if not production:
            css = self.target + '/ckan/ckan/public/base/css'
            if not exists(css + '/main.debug.css'):
                from shutil import copyfile
                copyfile(css + '/main.css', css + '/main.debug.css')

        ro = {
            self.target: '/project',
            scripts.get_script_path('datapusher.sh'): '/scripts/datapusher.sh'
        }

        if not is_boot2docker():
            ro[self.datadir + '/venv'] = '/usr/lib/ckan'

        datapusher = self.needs_datapusher()
        if datapusher:
            run_container(
                self._get_container_name('datapusher'),
                'datacats/web',
                '/scripts/datapusher.sh',
                ro=ro,
                volumes_from=(self._get_container_name('venv') if is_boot2docker() else None),
                log_syslog=log_syslog)

        while True:
            self._create_run_ini(port, production)
            try:
                self._run_web_container(port, command, address, log_syslog=log_syslog,
                                        datapusher=datapusher, interactive=interactive)
                if not is_boot2docker():
                    self.address = address
            except PortAllocatedError:
                port = self._next_port(port)
                continue
            break
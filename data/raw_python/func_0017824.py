def interactive_shell(self, command=None, paster=False, detach=False):
        """
        launch interactive shell session with all writable volumes

        :param: list of strings to execute instead of bash
        """
        if not exists(self.target + '/.bash_profile'):
            # this file is required for activating the virtualenv
            self.create_bash_profile()

        if not command:
            command = []
        use_tty = sys.stdin.isatty() and sys.stdout.isatty()

        background = environ.get('CIRCLECI', False) or detach

        if is_boot2docker():
            venv_volumes = ['--volumes-from', self._get_container_name('venv')]
        else:
            venv_volumes = ['-v', self.datadir + '/venv:/usr/lib/ckan:rw']

        self._create_run_ini(self.port, production=False, output='run.ini')
        self._create_run_ini(self.port, production=True, output='test.ini',
                             source='ckan/test-core.ini', override_site_url=False)

        script = scripts.get_script_path('shell.sh')
        if paster:
            script = scripts.get_script_path('paster.sh')
            if command and command != ['help'] and command != ['--help']:
                command += ['--config=/project/development.ini']
            command = [self.extension_dir] + command

        proxy_settings = self._proxy_settings()
        if proxy_settings:
            venv_volumes += ['-v',
                             self.sitedir + '/run/proxy-environment:/etc/environment:ro']

        links = {self._get_container_name('solr'): 'solr',
                 self._get_container_name('postgres'): 'db'}

        links.update({self._get_container_name(container): container for container
                      in self.extra_containers})

        link_params = []

        for link in links:
            link_params.append('--link')
            link_params.append(link + ':' + links[link])

        if 'datapusher' in self.containers_running():
            link_params.append('--link')
            link_params.append(self._get_container_name('datapusher') + ':datapusher')

        # FIXME: consider switching this to dockerpty
        # using subprocess for docker client's interactive session
        return subprocess.call([
            DOCKER_EXE, 'run',
            ] + (['--rm'] if not background else []) + [
            '-t' if use_tty else '',
            '-d' if detach else '-i',
            ] + venv_volumes + [
            '-v', self.target + ':/project:rw',
            '-v', self.sitedir + '/files:/var/www/storage:rw',
            '-v', script + ':/scripts/shell.sh:ro',
            '-v', scripts.get_script_path('paster_cd.sh') + ':/scripts/paster_cd.sh:ro',
            '-v', self.sitedir + '/run/run.ini:/project/development.ini:ro',
            '-v', self.sitedir +
                '/run/test.ini:/project/ckan/test-core.ini:ro'] +
            link_params +
            ['--hostname', self.name,
            'datacats/web', '/scripts/shell.sh'] + command)
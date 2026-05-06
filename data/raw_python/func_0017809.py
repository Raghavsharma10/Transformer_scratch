def create_ckan_ini(self):
        """
        Use make-config to generate an initial development.ini file
        """
        self.run_command(
            command='/scripts/run_as_user.sh /usr/lib/ckan/bin/paster make-config'
            ' ckan /project/development.ini',
            rw_project=True,
            ro={scripts.get_script_path('run_as_user.sh'): '/scripts/run_as_user.sh'},
            )
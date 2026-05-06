def update_ckan_ini(self, skin=True):
        """
        Use config-tool to update development.ini with our environment settings

        :param skin: use environment template skin plugin True/False
        """
        command = [
            '/usr/lib/ckan/bin/paster', '--plugin=ckan', 'config-tool',
            '/project/development.ini', '-e',
            'sqlalchemy.url = postgresql://<hidden>',
            'ckan.datastore.read_url = postgresql://<hidden>',
            'ckan.datastore.write_url = postgresql://<hidden>',
            'ckan.datapusher.url = http://datapusher:8800',
            'solr_url = http://solr:8080/solr',
            'ckan.storage_path = /var/www/storage',
            'ckan.plugins = datastore resource_proxy text_view ' +
            ('datapusher ' if exists(self.target + '/datapusher') else '')
            + 'recline_grid_view recline_graph_view'
            + (' {0}_theme'.format(self.name) if skin else ''),
            'ckan.site_title = ' + self.name,
            'ckan.site_logo =',
            'ckan.auth.create_user_via_web = false',
            ]
        self.run_command(command=command, rw_project=True)
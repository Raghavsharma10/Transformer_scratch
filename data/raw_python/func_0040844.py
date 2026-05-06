def kibana_config(self):
        """
        config kibana
        :return:
        """

        uncomment("/etc/kibana/kibana.yml", "#server.host:", use_sudo=True)
        sed('/etc/kibana/kibana.yml', 'server.host:.*',
            'server.host: "{0}"'.format(env.host_string), use_sudo=True)
        sudo('systemctl stop kibana.service')
        sudo('systemctl daemon-reload')
        sudo('systemctl enable kibana.service')
        sudo('systemctl start kibana.service')
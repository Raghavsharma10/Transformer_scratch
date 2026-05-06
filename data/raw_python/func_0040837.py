def systemctl_autostart(self, service_name, start_cmd, stop_cmd):
        """
        ubuntu 16.04 systemctl service config
        :param service_name:
        :param start_cmd:
        :param stop_cmd:
        :return:
        """
        # get config content
        service_content = bigdata_conf.systemctl_config.format(
            service_name=service_name,
            start_cmd=start_cmd,
            stop_cmd=stop_cmd
        )

        # write config into file
        with cd('/lib/systemd/system'):
            if not exists(service_name):
                sudo('touch {0}'.format(service_name))
            put(StringIO(service_content), service_name, use_sudo=True)

        # make service auto-start
        sudo('systemctl daemon-reload')
        sudo('systemctl disable {0}'.format(service_name))
        sudo('systemctl stop {0}'.format(service_name))
        sudo('systemctl enable {0}'.format(service_name))
        sudo('systemctl start {0}'.format(service_name))
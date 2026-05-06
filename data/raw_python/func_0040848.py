def reset_server_env(self, server_name, configure):
        """
        reset server env to server-name
        :param server_name:
        :param configure:
        :return:
        """
        env.host_string = configure[server_name]['host']
        env.user = configure[server_name]['user']
        env.password = configure[server_name]['passwd']
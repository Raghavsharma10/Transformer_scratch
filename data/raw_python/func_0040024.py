def login_server(self):
        """
        Login to server
        """
        local('ssh -i {0} {1}@{2}'.format(
            env.key_filename, env.user, env.host_string
        ))
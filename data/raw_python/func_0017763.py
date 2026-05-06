def create(self, environment, target_name):
        """
        Sends "create project" command to the remote server
        """
        remote_server_command(
            ["ssh", environment.deploy_target, "create", target_name],
            environment, self,
            clean_up=True,
            )
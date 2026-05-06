def admin_password(self, environment, target_name, password):
        """
        Return True if password was set successfully
        """
        try:
            remote_server_command(
                ["ssh", environment.deploy_target,
                    "admin_password", target_name, password],
                environment, self,
                clean_up=True
                )
            return True
        except WebCommandError:
            return False
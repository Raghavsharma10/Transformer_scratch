def deploy(self, environment, target_name, stream_output=None):
        """
        Return True if deployment was successful
        """
        try:
            remote_server_command(
                [
                    "rsync", "-lrv", "--safe-links", "--munge-links",
                    "--delete", "--inplace", "--chmod=ugo=rwX",
                    "--exclude=.datacats-environment",
                    "--exclude=.git",
                    "/project/.",
                    environment.deploy_target + ':' + target_name
                ],
                environment, self,
                include_project_dir=True,
                stream_output=stream_output,
                clean_up=True,
                )
        except WebCommandError as e:
            raise DatacatsError(
                "Unable to deploy `{0}` to remote server for some reason:\n"
                " datacats was not able to copy data to the remote server"
                .format((target_name,)),
                parent_exception=e
                )

        try:
            remote_server_command(
                [
                    "ssh", environment.deploy_target, "install", target_name,
                    ],
                environment, self,
                clean_up=True,
                )
            return True
        except WebCommandError as e:
            raise DatacatsError(
                "Unable to deploy `{0}` to remote server for some reason:\n"
                "datacats copied data to the server but failed to register\n"
                "(or `install`) the new catalog"
                .format((target_name,)),
                parent_exception=e
                )
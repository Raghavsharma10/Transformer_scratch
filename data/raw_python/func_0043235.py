def get_configs(self, command):
        """Compose a dictionary with information for writing the submit script."""

        logger.debug("Requesting one block with {} nodes per block and {} tasks per node".format(
            self.nodes_per_block, self.tasks_per_node))

        job_config = {}
        job_config["submit_script_dir"] = self.channel.script_dir
        job_config["nodes"] = self.nodes_per_block
        job_config["walltime"] = wtime_to_minutes(self.walltime)
        job_config["overrides"] = self.overrides
        job_config["user_script"] = command

        job_config["user_script"] = self.launcher(command,
                                                  self.tasks_per_node,
                                                  self.nodes_per_block)
        return job_config
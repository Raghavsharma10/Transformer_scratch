def submit(self, command, blocksize, job_name="parsl.auto"):
        """Submit the command as a slurm job of blocksize parallel elements.

        Parameters
        ----------
        command : str
            Command to be made on the remote side.
        blocksize : int
            Not implemented.
        job_name : str
            Name for the job (must be unique).

        Returns
        -------
        None or str
            If at capacity, returns None; otherwise, a string identifier for the job
        """

        if self.provisioned_blocks >= self.max_blocks:
            logger.warn("Slurm provider '{}' is at capacity (no more blocks will be added)".format(self.label))
            return None

        job_name = "{0}.{1}".format(job_name, time.time())

        script_path = "{0}/{1}.submit".format(self.script_dir, job_name)
        script_path = os.path.abspath(script_path)

        logger.debug("Requesting one block with {} nodes".format(self.nodes_per_block))

        job_config = {}
        job_config["submit_script_dir"] = self.channel.script_dir
        job_config["nodes"] = self.nodes_per_block
        job_config["tasks_per_node"] = self.tasks_per_node
        job_config["walltime"] = wtime_to_minutes(self.walltime)
        job_config["overrides"] = self.overrides
        job_config["partition"] = self.partition
        job_config["user_script"] = command

        # Wrap the command
        job_config["user_script"] = self.launcher(command,
                                                  self.tasks_per_node,
                                                  self.nodes_per_block)

        logger.debug("Writing submit script")
        self._write_submit_script(template_string, script_path, job_name, job_config)

        channel_script_path = self.channel.push_file(script_path, self.channel.script_dir)

        retcode, stdout, stderr = super().execute_wait("sbatch {0}".format(channel_script_path))

        job_id = None
        if retcode == 0:
            for line in stdout.split('\n'):
                if line.startswith("Submitted batch job"):
                    job_id = line.split("Submitted batch job")[1].strip()
                    self.resources[job_id] = {'job_id': job_id, 'status': 'PENDING', 'blocksize': blocksize}
        else:
            print("Submission of command to scale_out failed")
            logger.error("Retcode:%s STDOUT:%s STDERR:%s", retcode, stdout.strip(), stderr.strip())
        return job_id
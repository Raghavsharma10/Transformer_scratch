def submit(self, command="", blocksize=1, job_name="parsl.auto"):
        ''' The submit method takes the command string to be executed upon
        instantiation of a resource most often to start a pilot (such as IPP engine
        or even Swift-T engines).

        Args :
             - command (str) : The bash command string to be executed.
             - blocksize (int) : Blocksize to be requested

        KWargs:
             - job_name (str) : Human friendly name to be assigned to the job request

        Returns:
             - A job identifier, this could be an integer, string etc

        Raises:
             - ExecutionProviderException or its subclasses
        '''

        # Note: Fix this later to avoid confusing behavior.
        # We should always allocate blocks in integer counts of node_granularity
        if blocksize < self.nodes_per_block:
            blocksize = self.nodes_per_block

        # Set job name
        job_name = "{0}.{1}".format(job_name, time.time())

        # Set script path
        script_path = "{0}/{1}.submit".format(self.script_dir, job_name)
        script_path = os.path.abspath(script_path)

        job_config = self.get_configs(command, blocksize)

        logger.debug("Writing submit script")
        self._write_submit_script(template_string, script_path, job_name, job_config)

        channel_script_path = self.channel.push_file(script_path, self.channel.script_dir)
        cmd = "qsub -terse {0}".format(channel_script_path)
        retcode, stdout, stderr = super().execute_wait(cmd, 10)

        if retcode == 0:
            for line in stdout.split('\n'):
                job_id = line.strip()
                if not job_id:
                    continue
                self.resources[job_id] = {'job_id': job_id, 'status': 'PENDING', 'blocksize': blocksize}
                return job_id
        else:
            print("[WARNING!!] Submission of command to scale_out failed")
            logger.error("Retcode:%s STDOUT:%s STDERR:%s", retcode, stdout.strip(), stderr.strip())
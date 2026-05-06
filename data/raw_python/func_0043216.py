def submit(self, command, blocksize, job_name="parsl.auto"):
        ''' Submits the command onto an Local Resource Manager job of blocksize parallel elements.
        Submit returns an ID that corresponds to the task that was just submitted.

        If tasks_per_node <  1 : ! This is illegal. tasks_per_node should be integer

        If tasks_per_node == 1:
             A single node is provisioned

        If tasks_per_node >  1 :
             tasks_per_node * blocksize number of nodes are provisioned.

        Args:
             - command  :(String) Commandline invocation to be made on the remote side.
             - blocksize   :(float)

        Kwargs:
             - job_name (String): Name for job, must be unique

        Returns:
             - None: At capacity, cannot provision more
             - job_id: (string) Identifier for the job

        '''

        if self.provisioned_blocks >= self.max_blocks:
            logger.warn("[%s] at capacity, cannot add more blocks now", self.label)
            return None

        # Note: Fix this later to avoid confusing behavior.
        # We should always allocate blocks in integer counts of node_granularity
        if blocksize < self.nodes_per_block:
            blocksize = self.nodes_per_block

        # Set job name
        job_name = "parsl.{0}.{1}".format(job_name, time.time())

        # Set script path
        script_path = "{0}/{1}.submit".format(self.script_dir, job_name)
        script_path = os.path.abspath(script_path)

        logger.debug("Requesting blocksize:%s nodes_per_block:%s tasks_per_node:%s", blocksize, self.nodes_per_block,
                     self.tasks_per_node)

        job_config = {}
        # TODO : script_path might need to change to accommodate script dir set via channels
        job_config["submit_script_dir"] = self.channel.script_dir
        job_config["nodes"] = self.nodes_per_block
        job_config["task_blocks"] = self.nodes_per_block * self.tasks_per_node
        job_config["nodes_per_block"] = self.nodes_per_block
        job_config["tasks_per_node"] = self.tasks_per_node
        job_config["walltime"] = self.walltime
        job_config["overrides"] = self.overrides
        job_config["user_script"] = command

        # Wrap the command
        job_config["user_script"] = self.launcher(command,
                                                  self.tasks_per_node,
                                                  self.nodes_per_block)

        logger.debug("Writing submit script")
        self._write_submit_script(template_string, script_path, job_name, job_config)

        channel_script_path = self.channel.push_file(script_path, self.channel.script_dir)

        submit_options = ''
        if self.queue is not None:
            submit_options = '{0} -q {1}'.format(submit_options, self.queue)
        if self.account is not None:
            submit_options = '{0} -A {1}'.format(submit_options, self.account)

        launch_cmd = "qsub {0} {1}".format(submit_options, channel_script_path)
        retcode, stdout, stderr = self.channel.execute_wait(launch_cmd, 10)

        job_id = None
        if retcode == 0:
            for line in stdout.split('\n'):
                if line.strip():
                    job_id = line.strip()
                    self.resources[job_id] = {'job_id': job_id, 'status': 'PENDING', 'blocksize': blocksize}
        else:
            message = "Command '{}' failed with return code {}".format(launch_cmd, retcode)
            if (stdout is not None) and (stderr is not None):
                message += "\nstderr:{}\nstdout{}".format(stderr.strip(), stdout.strip())
            logger.error(message)

        return job_id
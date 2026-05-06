def submit(self, command, blocksize, job_name="parsl.auto"):
        ''' Submits the command onto an Local Resource Manager job of blocksize parallel elements.
        Submit returns an ID that corresponds to the task that was just submitted.

        If tasks_per_node <  1:
             1/tasks_per_node is provisioned

        If tasks_per_node == 1:
             A single node is provisioned

        If tasks_per_node >  1 :
             tasks_per_node * blocksize number of nodes are provisioned.

        Args:
             - command  :(String) Commandline invocation to be made on the remote side.
             - blocksize   :(float) - Not really used for local

        Kwargs:
             - job_name (String): Name for job, must be unique

        Returns:
             - None: At capacity, cannot provision more
             - job_id: (string) Identifier for the job

        '''

        job_name = "{0}.{1}".format(job_name, time.time())

        # Set script path
        script_path = "{0}/{1}.sh".format(self.script_dir, job_name)
        script_path = os.path.abspath(script_path)

        wrap_command = self.launcher(command, self.tasks_per_node, self.nodes_per_block)

        self._write_submit_script(wrap_command, script_path)

        job_id, proc = self.channel.execute_no_wait('bash {0}'.format(script_path), 3)
        self.resources[job_id] = {'job_id': job_id, 'status': 'RUNNING', 'blocksize': blocksize, 'proc': proc}

        return job_id
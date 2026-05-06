def submit(self, command='sleep 1', blocksize=1, job_name="parsl.auto"):
        """Submit command to an Azure instance.

        Submit returns an ID that corresponds to the task that was just submitted.

        Parameters
        ----------
        command : str
            Command to be invoked on the remote side.
        blocksize : int
            Number of blocks requested.
        job_name : str
             Prefix for job name.

        Returns
        -------
        None or str
            If at capacity (no more can be provisioned), None is returned. Otherwise,
            an identifier for the job is returned.
        """

        job_name = "parsl.auto.{0}".format(time.time())
        [instance, *rest] = self.deployer.deploy(command=command, job_name=job_name, blocksize=1)

        if not instance:
            logger.error("Failed to submit request to Azure")
            return None

        logger.debug("Started instance_id: {0}".format(instance.instance_id))

        state = translate_table.get(instance.state['Name'], "PENDING")

        self.resources[instance.instance_id] = {"job_id": instance.instance_id, "instance": instance, "status": state}

        return instance.instance_id
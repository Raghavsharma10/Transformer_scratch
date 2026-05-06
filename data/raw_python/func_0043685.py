def submit(self, command="", blocksize=1, job_name="parsl.auto"):
        ''' The submit method takes the command string to be executed upon
        instantiation of a resource most often to start a pilot.

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
        instance, name = self.create_instance(command=command)
        self.provisioned_blocks += 1
        self.resources[name] = {"job_id": name, "status": translate_table[instance['status']]}
        return name
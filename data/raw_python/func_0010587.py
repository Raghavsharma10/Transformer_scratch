def from_name(cls, name, *, queue=DefaultJobQueueName.Workflow,
                  clear_data_store=True, arguments=None):
        """ Create a workflow object from a workflow script.

        Args:
            name (str): The name of the workflow script.
            queue (str): Name of the queue the workflow should be scheduled to.
            clear_data_store (bool): Remove any documents created during the workflow
                                     run in the data store after the run.
            arguments (dict): Dictionary of additional arguments that are ingested
                              into the data store prior to the execution of the workflow.

        Returns:
            Workflow: A fully initialised workflow object
        """
        new_workflow = cls(queue=queue, clear_data_store=clear_data_store)
        new_workflow.load(name, arguments=arguments)
        return new_workflow
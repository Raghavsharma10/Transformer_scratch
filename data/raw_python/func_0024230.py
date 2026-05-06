def serialize_workflow(self):
        """
        Serializes the current WF.

        Returns:
            WF state data.
        """
        self.workflow.refresh_waiting_tasks()
        return CompactWorkflowSerializer().serialize_workflow(self.workflow,
                                                              include_spec=False)
def _clear_current_task(self):

        """
        Clear tasks related attributes, checks permissions
        While switching WF to WF, authentication and permissions are checked for new WF.
        """
        self.current.task_name = None
        self.current.task_type = None
        self.current.task = None
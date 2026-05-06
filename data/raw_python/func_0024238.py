def switch_to_external_wf(self):
        """
        External workflow switcher.

        This method copies main workflow information into
        a temporary dict `main_wf` and makes external workflow
        acting as main workflow.

        """

        # External WF name should be stated at main wf diagram and type should be service task.
        if (self.current.task_type == 'ServiceTask' and
                self.current.task.task_spec.type == 'external'):

            log.debug("Entering to EXTERNAL WF")

            # Main wf information is copied to main_wf.
            main_wf = self.wf_state.copy()

            # workflow name from main wf diagram is assigned to current workflow name.
            # workflow name must be either in task_data with key 'external_wf' or in main diagram's
            # topic.
            self.current.workflow_name = self.current.task_data.pop('external_wf', False) or self.\
                current.task.task_spec.topic

            # For external WF, check permission and authentication. But after cleaning current task.
            self._clear_current_task()

            # check for auth and perm. current task cleared, do against new workflow_name
            self.check_for_authentication()
            self.check_for_permission()

            # wf knowledge is taken for external wf.
            self.workflow_spec = self.get_worfklow_spec()
            # New WF instance is created for external wf.
            self.workflow = self.create_workflow()
            # Current WF is this WF instance.
            self.current.workflow = self.workflow
            # main_wf: main wf information.
            # in_external: it states external wf in progress.
            # finished: it shows that main wf didn't finish still progress in external wf.
            self.wf_state = {'main_wf': main_wf, 'in_external': True, 'finished': False}
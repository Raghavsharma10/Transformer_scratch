def run(self):
        """
        Main loop of the workflow engine

        - Updates ::class:`~WFCurrent` object.
        - Checks for Permissions.
        - Activates all READY tasks.
        - Runs referenced activities (method calls).
        - Saves WF states.
        - Stops if current task is a UserTask or EndTask.
        - Deletes state object if we finish the WF.

        """
        # FIXME: raise if first task after line change isn't a UserTask
        # FIXME: raise if last task of a workflow is a UserTask
        # actually this check should be done at parser
        is_lane_changed = False

        while self._should_we_run():
            self.check_for_rerun_user_task()
            task = None
            for task in self.workflow.get_tasks(state=Task.READY):
                self.current.old_lane = self.current.lane_name
                self.current._update_task(task)
                if self.catch_lane_change():
                    return
                self.check_for_permission()
                self.check_for_lane_permission()
                self.log_wf_state()
                self.switch_lang()
                self.run_activity()
                self.parse_workflow_messages()
                self.workflow.complete_task_from_id(self.current.task.id)
                self._save_or_delete_workflow()
                self.switch_to_external_wf()

            if task is None:
                break
        self.switch_from_external_to_main_wf()
        self.current.output['token'] = self.current.token

        # look for incoming ready task(s)
        for task in self.workflow.get_tasks(state=Task.READY):
            self.current._update_task(task)
            self.catch_lane_change()
            self.handle_wf_finalization()
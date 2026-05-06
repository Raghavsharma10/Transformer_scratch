def check_for_rerun_user_task(self):
        """
        Checks that the user task needs to re-run.
        If necessary, current task and pre task's states are changed and re-run.
        If wf_meta not in data(there is no user interaction from pre-task) and last completed task
        type is user task and current step is not EndEvent and there is no lane change,
        this user task is rerun.
        """
        data = self.current.input
        if 'wf_meta' in data:
            return

        current_task = self.workflow.get_tasks(Task.READY)[0]
        current_task_type = current_task.task_spec.__class__.__name__
        pre_task = current_task.parent
        pre_task_type = pre_task.task_spec.__class__.__name__

        if pre_task_type != 'UserTask':
            return

        if current_task_type == 'EndEvent':
            return

        pre_lane = pre_task.task_spec.lane
        current_lane = current_task.task_spec.lane
        if pre_lane == current_lane:
            pre_task._set_state(Task.READY)
            current_task._set_state(Task.MAYBE)
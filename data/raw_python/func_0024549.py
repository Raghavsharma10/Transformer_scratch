def create_tasks(self):
        """
        will create a WFInstance per object
        and per TaskInvitation for each role and WFInstance
        """
        roles = self.get_roles()

        if self.task_type in ["A", "D"]:
            instances = self.create_wf_instances(roles=roles)
            self.create_task_invitation(instances)

        elif self.task_type in ["C", "B"]:
            instances = self.create_wf_instances()
            self.create_task_invitation(instances, roles)
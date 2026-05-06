def send_workflow(self):
        """
        With the workflow instance and the task invitation is assigned a role.
        """
        task_invitation = TaskInvitation.objects.get(self.task_invitation_key)
        wfi = task_invitation.instance
        select_role = self.input['form']['select_role']
        if wfi.current_actor == self.current.role:
            task_invitation.role = RoleModel.objects.get(select_role)
            wfi.current_actor = RoleModel.objects.get(select_role)
            wfi.save()
            task_invitation.save()
            [inv.delete() for inv in TaskInvitation.objects.filter(instance=wfi) if
             not inv == task_invitation]
            title = _(u"Successful")
            msg = _(u"The workflow was assigned to someone else with success.")
        else:
            title = _(u"Unsuccessful")
            msg = _(u"This workflow does not belong to you, you cannot assign it to someone else.")

        self.current.msg_box(title=title, msg=msg)
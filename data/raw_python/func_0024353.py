def save_date(self):
        """
            Invitations with the same workflow status are deleted.
            Workflow instance and invitation roles change.

        """
        task_invitation = TaskInvitation.objects.get(self.task_invitation_key)
        wfi = task_invitation.instance
        if wfi.current_actor.exist and wfi.current_actor == self.current.role:

            dt_start = datetime.strptime(self.input['form']['start_date'], "%d.%m.%Y")
            dt_finish = datetime.strptime(self.input['form']['finish_date'], "%d.%m.%Y")

            task_invitation.start_date = dt_start
            task_invitation.finish_date = dt_finish
            task_invitation.save()

            wfi.start_date = dt_start
            wfi.finish_date = dt_finish
            wfi.save()

            title = _(u"Successful")
            msg = _(u"You've extended the workflow time.")
        else:
            title = _(u"Unsuccessful")
            msg = _(u"This workflow does not belong to you.")

        self.current.msg_box(title=title, msg=msg)
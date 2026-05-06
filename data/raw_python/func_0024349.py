def assign_yourself(self):
        """
            Assigning the workflow to itself.
            The selected job is checked to see if there is an assigned role.
            If it does not have a role assigned to it, it takes the job to itself
            and displays a message that the process is successful.
            If there is a role assigned to it, it does not do any operation
            and the message is displayed on the screen.

             .. code-block:: python

                #  request:
                   {
                   'task_inv_key': string,
                   }

        """
        task_invitation = TaskInvitation.objects.get(self.task_invitation_key)
        wfi = task_invitation.instance

        if not wfi.current_actor.exist:
            wfi.current_actor = self.current.role
            wfi.save()
            [inv.delete() for inv in TaskInvitation.objects.filter(instance=wfi) if
             not inv == task_invitation]
            title = _(u"Successful")
            msg = _(u"You have successfully assigned the job to yourself.")
        else:
            title = _(u"Unsuccessful")
            msg = _(u"Unfortunately, this job is already taken by someone else.")

        self.current.msg_box(title=title, msg=msg)
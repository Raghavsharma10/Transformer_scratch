def suspend(self):
        """
        If there is a role assigned to the workflow and
        it is the same as the user, it can drop the workflow.
        If it does not exist, it can not do anything.

            .. code-block:: python

                #  request:
                   {
                   'task_inv_key': string,
                   }

        """
        task_invitation = TaskInvitation.objects.get(self.task_invitation_key)
        wfi = task_invitation.instance

        if wfi.current_actor.exist and wfi.current_actor == self.current.role:
            for m in RoleModel.objects.filter(abstract_role=self.current.role.abstract_role,
                                              unit=self.current.role.unit):
                if m != self.current.role:
                    task_invitation.key = ''
                    task_invitation.role = m
                    task_invitation.save()

            wfi.current_actor = RoleModel()
            wfi.save()
            title = _(u"Successful")
            msg = _(u"You left the workflow.")
        else:
            title = _(u"Unsuccessful")
            msg = _(u"Unfortunately, this workflow does not belong to you or is already idle.")

        self.current.msg_box(title=title, msg=msg)
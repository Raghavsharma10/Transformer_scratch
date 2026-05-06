def delete_other_invitations(self):
        """
        When one person use an invitation, we should delete other invitations
        """
        # TODO: Signal logged-in users to remove the task from their task list
        self.objects.filter(instance=self.instance).exclude(key=self.key).delete()
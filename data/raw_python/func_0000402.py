def _log_action(self):
        """
        Adds a log entry for this action to the object history in the Django admin.
        """
        if self.publish_version == self.UNPUBLISH_CHOICE:
            message = 'Unpublished page (scheduled)'
        else:
            message = 'Published version {} (scheduled)'.format(self.publish_version)

        LogEntry.objects.log_action(
            user_id=self.user.pk,
            content_type_id=self.content_type.pk,
            object_id=self.object_id,
            object_repr=force_text(self.content_object),
            action_flag=CHANGE,
            change_message=message
        )
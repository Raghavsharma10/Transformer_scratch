def log(self, user, content, message):
        """creates a new log record

        :param user: user
        :param content: content instance
        :param message: change information
        """
        return self.create(
            user=user,
            content_type=ContentType.objects.get_for_model(content),
            object_id=content.pk,
            change_message=message
        )
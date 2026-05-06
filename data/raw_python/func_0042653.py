def delete(self, version):
        """
        Deletes the given version, not the object itself.
        No log entry is generated but the user is notified
        with a message.
        """
        # Shouldn't be able to delete live or draft version
        if version.state != version.DRAFT and \
                      version.state != version.PUBLISHED:
            version.delete()
            message = "%s version deleted." % version.date_published
            return self.write_message(message=message)
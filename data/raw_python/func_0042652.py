def revert(self, version, url):
        """
        Set the given version to be the active draft.
        This is done by calling the object's `make_draft` method.
        Logs the revert as a 'save' and messages the user.
        """
        message = "Draft replaced with %s version. This revert has not been published." % version.date_published
        version.make_draft()

        # Log action as a save
        self.log_action(self.object, CMSLog.SAVE, url=url)
        return self.write_message(message=message)
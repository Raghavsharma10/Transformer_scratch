def _publish(self):
        """
        Process a publish action on the related object, returns a boolean if a change is made.

        Only objects where a version change is needed will be updated.
        """
        obj = self.content_object
        version = self.get_version()
        actioned = False

        # Only update if needed
        if obj.current_version != version:
            version = self.get_version()
            obj.current_version = version
            obj.save(update_fields=['current_version'])
            actioned = True

        return actioned
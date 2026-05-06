def _unpublish(self):
        """
        Process an unpublish action on the related object, returns a boolean if a change is made.

        Only objects with a current active version will be updated.
        """
        obj = self.content_object
        actioned = False

        # Only update if needed
        if obj.current_version is not None:
            obj.current_version = None
            obj.save(update_fields=['current_version'])
            actioned = True

        return actioned
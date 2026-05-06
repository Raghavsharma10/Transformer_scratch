def get_version(self):
        """
        Get the version object for the related object.
        """
        return Version.objects.get(
            content_type=self.content_type,
            object_id=self.object_id,
            version_number=self.publish_version,
        )
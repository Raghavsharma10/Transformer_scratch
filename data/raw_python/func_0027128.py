def _is_version_duplicate(self):
        """ Define should new version be created for object or no.

            Reasons to provide custom check instead of default `ignore_revision_duplicates`:
             - no need to compare all revisions - it is OK if right object version exists in any revision;
             - need to compare object attributes (not serialized data) to avoid
               version creation on wrong <float> vs <int> comparison;
        """
        if self.id is None:
            return False
        try:
            latest_version = Version.objects.get_for_object(self).latest('revision__date_created')
        except Version.DoesNotExist:
            return False
        latest_version_object = latest_version._object_version.object
        fields = self.get_version_fields()
        return all([getattr(self, f) == getattr(latest_version_object, f) for f in fields])
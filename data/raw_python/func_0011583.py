def delete_all_versions(self, model_name, obj_pk):
        """Delete all versions of a cached instance."""
        if self.cache:
            for version in self.versions:
                key = self.key_for(version, model_name, obj_pk)
                self.cache.delete(key)
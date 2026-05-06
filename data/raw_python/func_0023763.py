def add_tags(self, tags):
        """
        Add tags to a server. Accepts tags as strings or Tag objects.
        """
        if self.cloud_manager.assign_tags(self.uuid, tags):
            tags = self.tags + [str(tag) for tag in tags]
            object.__setattr__(self, 'tags', tags)
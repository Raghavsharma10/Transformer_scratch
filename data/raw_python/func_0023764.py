def remove_tags(self, tags):
        """
        Add tags to a server. Accepts tags as strings or Tag objects.
        """
        if self.cloud_manager.remove_tags(self, tags):
            new_tags = [tag for tag in self.tags if tag not in tags]
            object.__setattr__(self, 'tags', new_tags)
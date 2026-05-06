def parent(self, parent_object, limit_parent_language=True):
        """
        Return all content items which are associated with a given parent object.
        """
        return self.all().parent(parent_object, limit_parent_language)
def tagsOf(self, obj):
        """
        Return an iterator of unicode strings - the tag names which apply to
        the given object.
        """
        return self.store.query(
            Tag,
            AND(Tag.catalog == self,
                Tag.object == obj)).getColumn("name")
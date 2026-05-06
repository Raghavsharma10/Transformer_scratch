def add_tag(self, tag, value):
        """ as tags are kept in a sorted order, a bisection is a fastest way to identify a correct position
        of or a new tag to be added. An additional check is required to make sure w don't add duplicates
        """
        index = bisect_left(self.tags, (tag, value))
        contains = False
        if index < len(self.tags):
            contains = self.tags[index] == (tag, value)
        if not contains:
            self.tags.insert(index, (tag, value))
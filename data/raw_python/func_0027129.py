def get_ancestors(self):
        """ Get all unique instance ancestors """
        ancestors = list(self.get_parents())
        ancestor_unique_attributes = set([(a.__class__, a.id) for a in ancestors])
        ancestors_with_parents = [a for a in ancestors if isinstance(a, DescendantMixin)]
        for ancestor in ancestors_with_parents:
            for parent in ancestor.get_ancestors():
                if (parent.__class__, parent.id) not in ancestor_unique_attributes:
                    ancestors.append(parent)
        return ancestors
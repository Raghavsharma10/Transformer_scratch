def is_ancestor_of(self, other, inclusive=False):
        """ class or instance level method which returns True if self is
        ancestor (closer to root) of other else False. Optional flag
        `inclusive` on whether or not to treat self as ancestor of self.

        For example see:

        * :mod:`sqlalchemy_mptt.tests.cases.integrity.test_hierarchy_structure`
        """
        if inclusive:
            return (self.tree_id == other.tree_id) \
                & (self.left <= other.left) \
                & (other.right <= self.right)
        return (self.tree_id == other.tree_id) \
            & (self.left < other.left) \
            & (other.right < self.right)
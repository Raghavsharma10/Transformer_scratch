def from_first_relation(cls, vertex0, vertex1):
        """Intialize a fresh match based on the first relation"""
        result = cls([(vertex0, vertex1)])
        result.previous_ends1 = set([vertex1])
        return result
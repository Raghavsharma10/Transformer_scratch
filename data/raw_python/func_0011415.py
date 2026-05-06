def _pfp__is_non_consecutive_duplicate(self, name, child):
        """Return True/False if the child is a non-consecutive duplicately named
        field. Consecutive duplicately-named fields are stored in an implicit array,
        non-consecutive duplicately named fields have a numeric suffix appended to their name"""

        if len(self._pfp__children) == 0:
            return False
         
        # it should be an implicit array
        if self._pfp__children[-1]._pfp__name == name:
            return False

        # if it's elsewhere in the children name map OR a collision sequence has already been
        # started for this name, it should have a numeric suffix
        # appended
        elif name in self._pfp__children_map or name in self._pfp__name_collisions:
            return True

        # else, no collision
        return False
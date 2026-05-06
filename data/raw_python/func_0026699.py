def check(self, val):
        """Make sure given value is consistent with this `Key` specification.

        NOTE: if `type` is 'None', then `listable` also is *not* checked.
        """
        # If there is no `type` requirement, everything is allowed
        if self.type is None:
            return True

        is_list = isinstance(val, list)
        # If lists are not allowed, and this is a list --> false
        if not self.listable and is_list:
            return False

        # `is_number` already checks for either list or single value
        if self.type == KEY_TYPES.NUMERIC and not is_number(val):
            return False
        elif (self.type == KEY_TYPES.TIME and
              not is_number(val) and '-' not in val and '/' not in val):
            return False
        elif self.type == KEY_TYPES.STRING:
            # If its a list, check first element
            if is_list:
                if not isinstance(val[0], basestring):
                    return False
            # Otherwise, check it
            elif not isinstance(val, basestring):
                return False
        elif self.type == KEY_TYPES.BOOL:
            if is_list and not isinstance(val[0], bool):
                return False
            elif not isinstance(val, bool):
                return False

        return True
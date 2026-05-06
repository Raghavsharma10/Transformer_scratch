def _find_id(self, result, uid):
        """
            This method performs a depth-first search for the given uid in the dictionary of results.
        """
        # if the result is a list
        if isinstance(result, list):
            # if the list has a valid entry
            if any([self._find_id(value, uid) for value in result]):
                # then we're done
                return True

        # otherwise results could be dictionaries
        if isinstance(result, dict):
            # the children of the result that are lists
            list_children = [value for value in result.values() if isinstance(value, list)]

            # go to every value that is a list
            for value in list_children:
                # if the value is a match
                if self._find_id(value, uid):
                    # we're done
                    return True

            # the children of the result that are dicts
            dict_children = [value for value in result.values() if isinstance(value, dict)]

            # perform the check on every child that is a dict
            for value in dict_children:
                # if the child is a match
                if self._find_id(value, uid):
                    # we're done
                    return True

            # if there are no values that are lists and there is an id key
            if not list_children and not dict_children and 'id' in result:
                # the value of the remote id field
                result_id = result['id']
                # we've found a match if the id field matches (cast to match type)
                return result_id == type(result_id)(uid)

        # we didn't find the result
        return False
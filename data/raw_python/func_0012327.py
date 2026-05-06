async def _check_for_matching_user(self, **user_filters):
        """
            This function checks if there is a user with the same uid in the
            remote user service
            Args:
                **kwds : the filters of the user to check for
            Returns:
                (bool): wether or not there is a matching user
        """
        # there is a matching user if there are no errors and no results from
        user_data = self._get_matching_user(user_filters)

        # return true if there were no errors and at lease one  result
        return not user_data['errors'] and len(user_data['data'][root_query()])
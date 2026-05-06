def update_user(self):
        """
        Save the state of the current user
        """
        # First create a copy of the current user
        user_dict = self.serialize()
        # Then delete the entities in the description field
        del user_dict['description']['entities']
        # Then upload user_dict
        user, meta = self._api.update_user('me', data=user_dict)
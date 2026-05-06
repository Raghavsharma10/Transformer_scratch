def set_action_token(self, type_=None, action_token=None):
        """Set action tokens

        type_ -- either "upload" or "image"
        action_token -- string obtained from user/get_action_token,
                        set None to remove the token
        """
        if action_token is None:
            del self._action_tokens[type_]
        else:
            self._action_tokens[type_] = action_token
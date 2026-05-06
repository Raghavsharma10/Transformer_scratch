def check_for_valid_postback_actions(self, required_keys, post_params):
        """
        checks if post_params contain required keys
        """
        for key in required_keys:
            if key not in post_params:
                return False
        return True
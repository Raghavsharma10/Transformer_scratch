def receive_update_post(self, post_params):
        """
        Update postbacks
        """

        if isinstance(post_params, dict):
            required_params = ['action', 'email', 'sig']
            if not self.check_for_valid_postback_actions(required_params, post_params):
                return False
        else:
            return False

        if post_params['action'] != 'update':
            return False

        signature = post_params['sig']
        post_params = post_params.copy()
        del post_params['sig']

        if signature != get_signature_hash(post_params, self.secret):
            return False

        return True
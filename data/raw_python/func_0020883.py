def receive_hardbounce_post(self, post_params):
        """
        Hard bounce postbacks
        """
        if isinstance(post_params, dict):
            required_params = ['action', 'email', 'sig']
            if not self.check_for_valid_postback_actions(required_params, post_params):
                return False
        else:
            return False

        if post_params['action'] != 'hardbounce':
            return False

        signature = post_params['sig']
        post_params = post_params.copy()
        del post_params['sig']

        if signature != get_signature_hash(post_params, self.secret):
            return False

        # for sends
        if 'send_id' in post_params:
            send_id = post_params['send_id']
            send_response = self.get_send(send_id)
            if not send_response.is_ok():
                return False
            send_obj = send_response.get_body()
            if not send_obj or 'email' not in send_obj:
                return False

        # for blasts
        if 'blast_id' in post_params:
            blast_id = post_params['blast_id']
            blast_response = self.get_blast(blast_id)
            if not blast_response.is_ok():
                return False
            blast_obj = blast_response.get_body()
            if not blast_obj:
                return False

        return True
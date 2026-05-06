def create_response(self, message=None, end_session=False, card_obj=None,
                        reprompt_message=None, is_ssml=None):
        """
        message - text message to be spoken out by the Echo
        end_session - flag to determine whether this interaction should end the session
        card_obj = JSON card object to substitute the 'card' field in the raw_response
        """
        response = dict(self.base_response)
        if message:
            response['response'] = self.create_speech(message, is_ssml)
        response['response']['shouldEndSession'] = end_session
        if card_obj:
            response['response']['card'] = card_obj
        if reprompt_message:
            response['response']['reprompt'] = self.create_speech(reprompt_message, is_ssml)
        return Response(response)
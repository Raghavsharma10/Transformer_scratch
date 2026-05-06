def _send_direct_message_new(self, messageobject: Dict[str, Dict]) -> Any:
        """
        :reference: https://developer.twitter.com/en/docs/direct-messages/sending-and-receiving/api-reference/new-event.html
        """
        headers, post_data = _buildmessageobject(messageobject)
        newdm_path = "/direct_messages/events/new.json"

        return tweepy.binder.bind_api(
            api=self.api,
            path=newdm_path,
            method="POST",
            require_auth=True,
        )(post_data=post_data, headers=headers)
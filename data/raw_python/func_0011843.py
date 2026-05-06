def send_dm_sos(self, message: str) -> None:
        """
        Send DM to owner if something happens.

        :param message: message to send to owner.
        :returns: None.
        """
        if self.owner_handle:
            try:
                # twitter changed the DM API and tweepy (as of 2019-03-08)
                # has not adapted.
                # fixing with
                # https://github.com/tweepy/tweepy/issues/1081#issuecomment-423486837
                owner_id = self.api.get_user(screen_name=self.owner_handle).id
                event = {
                    "event": {
                        "type": "message_create",
                        "message_create": {
                            "target": {
                                "recipient_id": f"{owner_id}",
                            },
                            "message_data": {
                                "text": message
                            }
                        }
                    }
                }

                self._send_direct_message_new(event)

            except tweepy.TweepError as de:
                self.lerror(f"Error trying to send DM about error!: {de}")

        else:
            self.lerror("Can't send DM SOS, no owner handle.")
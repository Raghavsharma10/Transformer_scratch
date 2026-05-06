def from_http(
        cls,
        raw_body: MutableMapping,
        verification_token: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> "Event":
        """
        Create an event with data coming from the HTTP Event API.

        If the event type is a message a :class:`slack.events.Message` is returned.

        Args:
            raw_body: Decoded body of the Event API request
            verification_token: Slack verification token used to verify the request came from slack
            team_id: Verify the event is for the correct team

        Returns:
            :class:`slack.events.Event` or :class:`slack.events.Message`

        Raises:
            :class:`slack.exceptions.FailedVerification`: when `verification_token` or `team_id` does not match the
                                                          incoming event's.
        """
        if verification_token and raw_body["token"] != verification_token:
            raise exceptions.FailedVerification(raw_body["token"], raw_body["team_id"])

        if team_id and raw_body["team_id"] != team_id:
            raise exceptions.FailedVerification(raw_body["token"], raw_body["team_id"])

        if raw_body["event"]["type"].startswith("message"):
            return Message(raw_body["event"], metadata=raw_body)
        else:
            return Event(raw_body["event"], metadata=raw_body)
def handle_error(
            self,
            *,
            message: str,
            error: tweepy.TweepError,
    ) -> OutputRecord:
        """
        Handle error while trying to do something.

        :param message: message to send in DM regarding error.
        :param e: tweepy error object.
        :returns: OutputRecord containing an error.
        """
        self.lerror(f"Got an error! {error}")

        # Handle errors if we know how.
        try:
            code = error[0]["code"]
            if code in self.handled_errors:
                self.handled_errors[code]
            else:
                self.send_dm_sos(message)

        except Exception:
            self.send_dm_sos(message)

        return TweetRecord(error=error)
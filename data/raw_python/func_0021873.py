def response(self, in_thread: Optional[bool] = None) -> "Message":
        """
        Create a response message.

        Depending on the incoming message the response can be in a thread. By default the response follow where the
        incoming message was posted.

        Args:
            in_thread (boolean): Overwrite the `threading` behaviour

        Returns:
             a new :class:`slack.event.Message`
        """
        data = {"channel": self["channel"]}

        if in_thread:
            if "message" in self:
                data["thread_ts"] = (
                    self["message"].get("thread_ts") or self["message"]["ts"]
                )
            else:
                data["thread_ts"] = self.get("thread_ts") or self["ts"]
        elif in_thread is None:
            if "message" in self and "thread_ts" in self["message"]:
                data["thread_ts"] = self["message"]["thread_ts"]
            elif "thread_ts" in self:
                data["thread_ts"] = self["thread_ts"]

        return Message(data)
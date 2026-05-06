def serialize(self) -> dict:
        """
        Serialize the message for sending to slack API

        Returns:
            serialized message
        """
        data = {**self}
        if "attachments" in self:
            data["attachments"] = json.dumps(self["attachments"])
        return data
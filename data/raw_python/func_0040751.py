def __make_message(self, topic, content):
        """
        Prepares the message content
        """
        return {"uid": str(uuid.uuid4()).replace('-', '').upper(),
                "topic": topic,
                "content": content}
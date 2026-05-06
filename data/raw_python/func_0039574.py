def put_content(self, content):
        """
        Publishes a message straight to SNS.

        :param bytes content: raw bytes content to publish, will decode to ``UTF-8`` if string is detected
        """
        if not isinstance(content, str):
            content = content.decode('utf-8')

        self.topic.publish(Message=content, **self.storage_args)
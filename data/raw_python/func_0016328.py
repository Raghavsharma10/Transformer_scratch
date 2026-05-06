def attach_binary(self, content, filename):
        """
        Attaches given binary data.

        :param bytes content: Binary data to be attached.
        :param str filename:
        :return: None.
        """
        content_type = guess_content_type(filename)
        payload = {"Name": filename, "Content": b64encode(content).decode("utf-8"), "ContentType": content_type}
        self.attach(payload)
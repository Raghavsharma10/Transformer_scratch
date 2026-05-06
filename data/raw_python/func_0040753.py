def send(self, target, topic, content):
        """
        Fires a message
        """
        event = threading.Event()
        results = []

        def got_message(sender, content):
            results.append(content)
            event.set()

        self.post(target, topic, content, got_message)
        event.wait()

        return results
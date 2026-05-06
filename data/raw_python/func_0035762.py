def put_text(self, key, text):
        """Put the text into the storage associated with the key."""
        with open(key, "w") as fh:
            fh.write(text)
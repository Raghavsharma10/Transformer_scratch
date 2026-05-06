def exists(self):
        """Uses ``HEAD`` requests for efficiency."""

        try:
            self.blob.reload()
            return True
        except google.cloud.exceptions.NotFound: return False
def exists(self):
        """Uses ``HEAD`` requests for efficiency."""

        try:
            self.s3_object.load()
            return True
        except botocore.exceptions.ClientError: return False
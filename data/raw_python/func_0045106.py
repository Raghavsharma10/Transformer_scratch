def complete(self, return_code):
        """
        Mark the process as complete with provided return_code
        """
        self.return_code = return_code
        self.status = 'COMPLETE' if not return_code else 'FAILED'
        self.end_time = datetime.datetime.now()
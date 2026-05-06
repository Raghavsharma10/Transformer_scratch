def save_last_response_to_file(self, filename):
        """Saves the body of the last response to a file

        @param filename: Filename to save to
        @return: Returns False if there is an OS error, True if successful
        """
        response = self.get_last_response()
        return self.save_response_to_file(response, filename)
def json_success(self, json):
        """
        Check the JSON response object for the success flag

        Parameters
        ----------
        json : dict
            A dictionary representing a JSON object from lendingclub.com
        """
        if type(json) is dict and 'result' in json and json['result'] == 'success':
            return True
        return False
def get_response(self, url, params={}, method="get"):
        """
        It will return json response based on given url, params and methods.
    
        Arg:    
           params: 'dictionary'
           url: 'url' format
           method: default 'get', support method 'post' 
        Return:
           json data    
        """

        if method == "post":
            response_data = json.loads(requests.post(url, params=params).text)
        else:
            params["access_token"] = self.get_active_token()
            response_data = json.loads(requests.get(url, params=params).text)

        if not response_data['status_code'] == 200:
            if "status_msg" in response_data:
                logger.error("Bad response: " + response_data['status_msg'])
            else:
                logger.error("Some thing went wrong, please check your " + \
                             "request params Example: card_type and date")

        return response_data
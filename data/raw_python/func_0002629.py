def make_request(cls, url, method, params=None, basic_auth=None, timeout=600):
        """ Makes a cURL POST request to the given URL, specifying the data to be passed in as
         {"method": method, "params": parameters}

        :param str url: URL to connect to.
        :param str method: The API method to call.
        :param dict params: Dictionary object of the parameters associated with the `method` given. None by default.
        :param list | tuple basic_auth: List containing your username and password as ['username', 'password'].
         This is empty by default, however it is required by all of the `lbrycrd` methods
        :param float timeout: Amount of seconds to wait for the server's response before we timeout.
        :raises LBRYException: If the request returns an error when calling the API
        :return: A `dict` of the JSON result member of the request
        :rtype: dict, PreparedResponse
        """

        # Default parameters
        params = {} if params is None else params

        # Increment the request ID
        cls.request_id += 1

        # Weed out all the None valued params
        params = {k: v for (k, v) in params.items() if v is not None}

        # This is the data to be sent
        data = {"method": method, "params": params, "jsonrpc": "2.0", "id": cls.request_id}

        headers = {"Content-Type": "application/json-rpc",  # sends the request as a json
                   "user-agent": "LBRY python3-api"}    # Sets the user agent

        # You could create a request object and then make a prepared request object
        # And then be able to print the Request that will be sent
        request = requests.Request('POST', url, json=data, headers=headers, auth=basic_auth)

        prepared = request.prepare()

        try:

            # Create a session object
            sesh = requests.Session()

            # Send the prepared request object through
            response = sesh.send(prepared, timeout=timeout)

            response_json = response.json()

            # Successful request was made
            if 'result' in response_json:

                # Returns the Result sub-JSON formatted as a dict
                return response_json['result'], response

            # If the response we received from the LBRY http post had an error
            elif 'error' in response_json:
                raise LBRYUtils.LBRYException("POST Request made to LBRY received an error",
                                              response_json, response.status_code, prepared)

        except requests.HTTPError as HE:
            print(HE)

            return None, None

        except requests.RequestException as RE:
            # Print the Request Exception given
            print(RE)

            print("Printing Request Created:\n")

            LBRYUtils.print_request(prepared)

            return None, None
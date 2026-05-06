def send_request(self, send_function=None):
        """
        Sends the assembled request on the child object.
        @type send_function: function reference
        @keyword send_function: A function reference (passed without the
            parenthesis) to a function that will send the request. This
            allows for overriding the default function in cases such as
            validation requests.
        """

        # Send the request and get the response back.
        try:
            # If the user has overridden the send function, use theirs
            # instead of the default.
            if send_function:
                # Follow the overridden function.
                self.response = send_function()
            else:
                # Default scenario, business as usual.
                self.response = self._assemble_and_send_request()
        except suds.WebFault as fault:
            # When this happens, throw an informative message reminding the
            # user to check all required variables, making sure they are
            # populated and valid
            raise SchemaValidationError(fault.fault)

        # Check the response for general Fedex errors/failures that aren't
        # specific to any given WSDL/request.
        self.__check_response_for_fedex_error()

        # Check the response for errors specific to the particular request.
        # This method can be overridden by a method on the child class object.
        self._check_response_for_request_errors()

        # Check the response for errors specific to the particular request.
        # This method can be overridden by a method on the child class object.
        self._check_response_for_request_warnings()

        # Debug output. (See Request and Response output)
        self.logger.debug("== FEDEX QUERY RESULT ==")
        self.logger.debug(self.response)
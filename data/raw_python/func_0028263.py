def create(cls, endpoint_name, json_body, original_response):
        """Factory for creating the correct type of Response based on the data.
        Args:
            endpoint_name (str) - The endpoint of the request, such as "property/value"
            json_body - The response body in json format.
            original_response (response object) - server response returned from an http request.
        """

        if endpoint_name == "property/value_report":
            return ValueReportResponse(endpoint_name, json_body, original_response)

        if endpoint_name == "property/rental_report":
            return RentalReportResponse(endpoint_name, json_body, original_response)

        prefix = endpoint_name.split("/")[0]

        if prefix == "block":
            return BlockResponse(endpoint_name, json_body, original_response)

        if prefix == "zip":
            return ZipCodeResponse(endpoint_name, json_body, original_response)

        if prefix == "msa":
            return MsaResponse(endpoint_name, json_body, original_response)

        return PropertyResponse(endpoint_name, json_body, original_response)
def search_items(self, mode, request):
        """Invoke query/scan by name.

        Response always includes "Count" and "ScannedCount"

        :param str mode: "query" or "scan"
        :param request: Unpacked into :func:`boto3.DynamoDB.Client.query` or :func:`boto3.DynamoDB.Client.scan`
        """
        validate_search_mode(mode)
        method = getattr(self.dynamodb_client, mode)
        try:
            response = method(**request)
        except botocore.exceptions.ClientError as error:
            raise BloopException("Unexpected error during {}.".format(mode)) from error
        standardize_query_response(response)
        return response
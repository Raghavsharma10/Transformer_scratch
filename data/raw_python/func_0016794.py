def transaction_write(self, items, client_request_token):
        """
        Wraps :func:`boto3.DynamoDB.Client.db.transact_write_items`.

        :param items: Unpacked into "TransactionItems" for :func:`boto3.DynamoDB.Client.transact_write_items`
        :param client_request_token: Idempotency token valid for 10 minutes from first use.
            Unpacked into "ClientRequestToken"
        :raises bloop.exceptions.TransactionCanceled: if the transaction was canceled.
        """
        try:
            self.dynamodb_client.transact_write_items(
                TransactItems=items,
                ClientRequestToken=client_request_token
            )
        except botocore.exceptions.ClientError as error:
            if error.response["Error"]["Code"] == "TransactionCanceledException":
                raise TransactionCanceled from error
            raise BloopException("Unexpected error during transaction write.") from error
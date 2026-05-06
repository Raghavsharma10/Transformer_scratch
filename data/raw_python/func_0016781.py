def save_item(self, item):
        """Save an object to DynamoDB.

        :param item: Unpacked into kwargs for :func:`boto3.DynamoDB.Client.update_item`.
        :raises bloop.exceptions.ConstraintViolation: if the condition (or atomic) is not met.
        """
        try:
            self.dynamodb_client.update_item(**item)
        except botocore.exceptions.ClientError as error:
            handle_constraint_violation(error)
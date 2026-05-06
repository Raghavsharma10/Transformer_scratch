def delete_item(self, item):
        """Delete an object in DynamoDB.

        :param item: Unpacked into kwargs for :func:`boto3.DynamoDB.Client.delete_item`.
        :raises bloop.exceptions.ConstraintViolation: if the condition (or atomic) is not met.
        """
        try:
            self.dynamodb_client.delete_item(**item)
        except botocore.exceptions.ClientError as error:
            handle_constraint_violation(error)
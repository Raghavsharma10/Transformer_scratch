def load_items(self, items):
        """Loads any number of items in chunks, handling continuation tokens.

        :param items: Unpacked in chunks into "RequestItems" for :func:`boto3.DynamoDB.Client.batch_get_item`.
        """
        loaded_items = {}
        requests = collections.deque(create_batch_get_chunks(items))
        while requests:
            request = requests.pop()
            try:
                response = self.dynamodb_client.batch_get_item(RequestItems=request)
            except botocore.exceptions.ClientError as error:
                raise BloopException("Unexpected error while loading items.") from error

            # Accumulate results
            for table_name, table_items in response.get("Responses", {}).items():
                loaded_items.setdefault(table_name, []).extend(table_items)

            # Push additional request onto the deque.
            # "UnprocessedKeys" is {} if this request is done
            if response["UnprocessedKeys"]:
                requests.append(response["UnprocessedKeys"])
        return loaded_items
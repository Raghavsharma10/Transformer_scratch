def _search(self, search_terms, begins_with=None):
        """
        Returns a list of Archive id's in the table on Dynamo

        """

        kwargs = dict(
            ProjectionExpression='#id',
            ExpressionAttributeNames={"#id": "_id"})

        if len(search_terms) > 0:
            kwargs['FilterExpression'] = reduce(
                lambda x, y: x & y,
                [Attr('tags').contains(arg) for arg in search_terms])

        if begins_with:
            if 'FilterExpression' in kwargs:
                kwargs['FilterExpression'] = kwargs[
                    'FilterExpression'] & Key('_id').begins_with(begins_with)

            else:
                kwargs['FilterExpression'] = Key(
                    '_id').begins_with(begins_with)

        while True:
            res = self._table.scan(**kwargs)
            for r in res['Items']:
                yield r['_id']
            if 'LastEvaluatedKey' in res:
                kwargs['ExclusiveStartKey'] = res['LastEvaluatedKey']
            else:
                break
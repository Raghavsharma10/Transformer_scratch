def retrieve(self, request, _id):
        """
        Returns the document containing the given _id or 404
        """
        _id = deserialize(_id)

        retrieved = self.collection.find_one({'_id': _id})
        if retrieved:
            return Response(serialize(retrieved))
        else:
            return Response(
                response=serialize(
                    DocumentNotFoundError(self.collection.__name__, _id)
                ),
                status=400
            )
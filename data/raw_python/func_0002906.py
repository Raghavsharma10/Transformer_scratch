def delete(self, request, _id):
        """
        Deletes the document with the given _id if it exists
        """
        _id = deserialize(_id)

        to_delete = self.collection.get({'_id': _id})
        if to_delete:
            deleted = to_delete.delete()

            return Response(
                response=serialize(deleted),
                status=(
                    200 if not all(
                        key in deleted for key in [
                            'error_code', 'error_type', 'error_message'
                        ]
                    ) else 400
                )
            )
        else:
            return Response(
                response=serialize(
                    DocumentNotFoundError(self.collection.__name__, _id)
                ),
                status=404
            )
def update(self, request, _id):
        """
        Updates the document with the given _id using the given data
        """
        _id = deserialize(_id)

        to_update = self.collection.find_one({'_id': _id})
        if to_update:
            document = self.collection(dict(to_update, **request.json))
            document.updated_at = datetime.utcnow()

            updated = document.update()
            return Response(
                response=serialize(updated),
                status=(
                    200 if not all(
                        key in updated for key in [
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
                status=400
            )
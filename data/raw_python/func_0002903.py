def create(self, request):
        """
        Creates a new document based on the given data
        """
        document = self.collection(request.json)
        document.created_at = datetime.utcnow()
        document.updated_at = document.created_at

        created = document.insert()
        return Response(
            response=serialize(created),
            status=(
                201 if not all(
                    key in created for key in [
                        'error_code', 'error_type', 'error_message'
                    ]
                ) else 400
            )
        )
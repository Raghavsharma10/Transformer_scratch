def fetch_object(self, doc_id):
        """Fetch the document by its PK."""
        try:
            return self.object_class.objects.get(pk=doc_id)
        except self.object_class.DoesNotExist:
            raise ReferenceNotFoundError
def delete(self, request, pzone_pk, operation_pk):
        """Remove an operation from the given pzone."""

        # note : we're not using the pzone_pk here since it's not actually
        #   necessary for getting an operation by pk, but it sure makes the urls
        #   nicer!

        # attempt to delete operation
        try:
            operation = PZoneOperation.objects.get(pk=operation_pk)
        except PZoneOperation.DoesNotExist:
            raise Http404("Cannot find given operation.")

        # delete operation
        operation.delete()

        # successful delete, return 204
        return Response("", 204)
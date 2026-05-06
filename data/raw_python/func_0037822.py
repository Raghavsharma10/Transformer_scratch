def get(self, request, pzone_pk):
        """Get all the operations for a given pzone."""

        # attempt to get given pzone
        try:
            pzone = PZone.objects.get(pk=pzone_pk)
        except PZone.DoesNotExist:
            raise Http404("Cannot find given pzone.")

        # bulid filters
        filters = {"pzone": pzone}

        if "from" in request.GET:
            parsed = dateparse.parse_datetime(request.GET["from"])
            if parsed is not None:
                filters["when__gte"] = parsed

        if "to" in request.GET:
            parsed = dateparse.parse_datetime(request.GET["to"])
            if parsed is not None:
                filters["when__lt"] = parsed

        # get operations and serialize them
        operations = PZoneOperation.objects.filter(**filters)

        # return a json response with serialized operations
        return Response(self.serialize_operations(operations), content_type="application/json")
def post(self, request, pzone_pk):
        """Add a new operation to the given pzone, return json of the new operation."""

        # attempt to get given content list
        pzone = None
        try:
            pzone = PZone.objects.get(pk=pzone_pk)
        except PZone.DoesNotExist:
            raise Http404("Cannot find given pzone.")

        json_obj = []
        http_status = 500

        json_op = json.loads(request.body.decode("utf8"))
        if not isinstance(json_op, list):
            json_op = [json_op]

        for data in json_op:
            try:
                serializer = self.get_serializer_class_by_name(data["type_name"])
            except ContentType.DoesNotExist as e:
                json_obj = {"errors": [str(e)]}
                http_status = 400
                break

            serialized = serializer(data=data)
            if serialized.is_valid():
                # object is valid, save it
                serialized.save()

                # set response data
                json_obj.append(serialized.data)
                http_status = 200
            else:
                # object is not valid, return errors in a 400 response
                json_obj = serialized.errors
                http_status = 400
                break

        if http_status == 200 and len(json_obj) == 1:
            json_obj = json_obj[0]

        # cache the time in seconds until the next operation occurs
        next_ops = PZoneOperation.objects.filter(when__lte=timezone.now())
        if len(next_ops) > 0:
            # we have at least one operation, ordered soonest first
            next_op = next_ops[0]
            # cache with expiry number of seconds until op should exec
            cache.set('pzone-operation-expiry-' + pzone.name, next_op.when, 60 * 60 * 5)

        return Response(
            json_obj,
            status=http_status,
            content_type="application/json"
        )
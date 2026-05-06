def retrieve(self, request, *args, **kwargs):
        """Retrieve pzone as a preview or applied if no preview is provided."""

        when_param = get_query_params(self.request).get("preview", None)
        pk = self.kwargs["pk"]

        when = None
        if when_param:
            try:
                when = parse_date(when_param)
            except ValueError:
                # invalid format, set back to None
                when = None

        pzone = None
        if when:
            # we have a date, use it
            pzone = PZone.objects.preview(pk=pk, when=when)
        else:
            # we have no date, just get the pzone
            pzone = PZone.objects.applied(pk=pk)

        # turn content list into json
        return Response(PZoneSerializer(pzone).data, content_type="application/json")
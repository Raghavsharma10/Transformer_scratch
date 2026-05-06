def publish(self, request, **kwargs):
        """sets the `published` value of the `Content`

        :param request: a WSGI request object
        :param kwargs: keyword arguments (optional)
        :return: `rest_framework.response.Response`
        """
        content = self.get_object()

        if "published" in get_request_data(request):
            if not get_request_data(request)["published"]:
                content.published = None
            else:
                publish_dt = parse_datetime(get_request_data(request)["published"])
                if publish_dt:
                    publish_dt = publish_dt.astimezone(timezone.utc)
                else:
                    publish_dt = None
                content.published = publish_dt
        else:
            content.published = timezone.now()

        content.save()
        LogEntry.objects.log(request.user, content, content.get_status())
        return Response({"status": content.get_status(), "published": content.published})
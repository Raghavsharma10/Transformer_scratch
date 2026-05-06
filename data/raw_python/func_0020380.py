def update(self, request, key):
        """Set an email address as primary address."""

        request.UPDATE = http.QueryDict(request.body)

        email_addr = request.UPDATE.get('email')
        user_id = request.UPDATE.get('user')
        if not email_addr:
            return http.HttpResponseBadRequest()

        try:
            email = EmailAddress.objects.get(address=email_addr,
                                             user_id=user_id)
        except EmailAddress.DoesNotExist:
            raise http.Http404

        email.user.email = email_addr
        email.user.save()
        return http.HttpResponse(status=204)
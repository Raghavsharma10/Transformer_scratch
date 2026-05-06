def delete(self, request, key):
        """Remove an email address, validated or not."""

        request.DELETE = http.QueryDict(request.body)
        email_addr = request.DELETE.get('email')
        user_id = request.DELETE.get('user')

        if not email_addr:
            return http.HttpResponseBadRequest()

        try:
            email = EmailAddressValidation.objects.get(address=email_addr,
                                                       user_id=user_id)
        except EmailAddressValidation.DoesNotExist:
            pass
        else:
            email.delete()
            return http.HttpResponse(status=204)

        try:
            email = EmailAddress.objects.get(address=email_addr,
                                             user_id=user_id)
        except EmailAddress.DoesNotExist:
            raise http.Http404

        email.user = None
        email.save()
        return http.HttpResponse(status=204)
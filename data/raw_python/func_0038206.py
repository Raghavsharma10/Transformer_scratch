def retrieve(self, request, *args, **kwargs):
        """gets basic information about the user

        :param request: a WSGI request object
        :param args: inline arguments (optional)
        :param kwargs: keyword arguments (optional)
        :return: `rest_framework.response.Response`
        """
        data = UserSerializer().to_representation(request.user)

        # add superuser flag only if user is a superuser, putting it here so users can only
        # tell if they are themselves superusers
        if request.user.is_superuser:
            data['is_superuser'] = True

        # attempt to add a firebase token if we have a firebase secret
        secret = getattr(settings, 'FIREBASE_SECRET', None)
        if secret:
            # use firebase auth to provide auth variables to firebase security api
            firebase_auth_payload = {
                'id': request.user.pk,
                'username': request.user.username,
                'email': request.user.email,
                'is_staff': request.user.is_staff
            }
            data['firebase_token'] = create_token(secret, firebase_auth_payload)

        return Response(data)
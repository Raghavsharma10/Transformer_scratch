def password(self, request, uuid=None):
        """
        To change a user password, submit a **POST** request to the user's RPC URL, specifying new password
        by staff user or account owner.

        Password is expected to be at least 7 symbols long and contain at least one number
        and at least one lower or upper case.

        Example of a valid request:

        .. code-block:: http

            POST /api/users/e0c058d06864441fb4f1c40dee5dd4fd/password/ HTTP/1.1
            Content-Type: application/json
            Accept: application/json
            Authorization: Token c84d653b9ec92c6cbac41c706593e66f567a7fa4
            Host: example.com

            {
                "password": "nQvqHzeP123",
            }
        """
        user = self.get_object()

        serializer = serializers.PasswordSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        new_password = serializer.validated_data['password']
        user.set_password(new_password)
        user.save()

        return Response({'detail': _('Password has been successfully updated.')},
                        status=status.HTTP_200_OK)
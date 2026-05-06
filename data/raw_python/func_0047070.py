def pre_dispatch(self, request, path_args):
        """
        Pre dispatch hook
        """
        secret_key = self.get_secret_key(request, path_args)
        if not secret_key:
            raise PermissionDenied('Signature not valid.')

        try:
            signing.verify_url_path(request.path, request.GET, secret_key)
        except SigningError as ex:
            raise PermissionDenied(str(ex))
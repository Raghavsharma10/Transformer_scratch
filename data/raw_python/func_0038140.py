def create_from_request(self, request):
        """
        Generate a token from the OAuth callback request. Must contain 'code' in GET.
        :param request: OAuth callback request.
        :return: :class:`esi.models.Token`
        """
        logger.debug("Creating new token for {0} session {1}".format(request.user, request.session.session_key[:5]))
        code = request.GET.get('code')
        # attach a user during creation for some functionality in a post_save created receiver I'm working on elsewhere
        model = self.create_from_code(code, user=request.user if request.user.is_authenticated else None)
        return model
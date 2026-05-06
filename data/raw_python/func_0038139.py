def create_from_code(self, code, user=None):
        """
        Perform OAuth code exchange to retrieve a token.
        :param code: OAuth grant code.
        :param user: User who will own token.
        :return: :class:`esi.models.Token`
        """

        # perform code exchange
        logger.debug("Creating new token from code {0}".format(code[:-5]))
        oauth = OAuth2Session(app_settings.ESI_SSO_CLIENT_ID, redirect_uri=app_settings.ESI_SSO_CALLBACK_URL)
        token = oauth.fetch_token(app_settings.ESI_TOKEN_URL, client_secret=app_settings.ESI_SSO_CLIENT_SECRET,
                                  code=code)
        r = oauth.request('get', app_settings.ESI_TOKEN_VERIFY_URL)
        r.raise_for_status()
        token_data = r.json()
        logger.debug(token_data)

        # translate returned data to a model
        model = self.create(
            character_id=token_data['CharacterID'],
            character_name=token_data['CharacterName'],
            character_owner_hash=token_data['CharacterOwnerHash'],
            access_token=token['access_token'],
            refresh_token=token['refresh_token'],
            token_type=token_data['TokenType'],
            user=user,
        )

        # parse scopes
        if 'Scopes' in token_data:
            from esi.models import Scope
            for s in token_data['Scopes'].split():
                try:
                    scope = Scope.objects.get(name=s)
                    model.scopes.add(scope)
                except Scope.DoesNotExist:
                    # This scope isn't included in a data migration. Create a placeholder until it updates.
                    try:
                        help_text = s.split('.')[1].replace('_', ' ').capitalize()
                    except IndexError:
                        # Unusual scope name, missing periods.
                        help_text = s.replace('_', ' ').capitalize()
                    scope = Scope.objects.create(name=s, help_text=help_text)
                    model.scopes.add(scope)
            logger.debug("Added {0} scopes to new token.".format(model.scopes.all().count()))

        if not app_settings.ESI_ALWAYS_CREATE_TOKEN:
            # see if we already have a token for this character and scope combination
            # if so, we don't need a new one
            queryset = self.get_queryset().equivalent_to(model)
            if queryset.exists():
                logger.debug(
                    "Identified {0} tokens equivalent to new token. Updating access and refresh tokens.".format(
                        queryset.count()))
                queryset.update(
                    access_token=model.access_token,
                    refresh_token=model.refresh_token,
                    created=model.created,
                )
                if queryset.filter(user=model.user).exists():
                    logger.debug("Equivalent token with same user exists. Deleting new token.")
                    model.delete()
                    model = queryset.filter(user=model.user)[0]  # pick one at random

        logger.debug("Successfully created {0} for user {1}".format(repr(model), user))
        return model
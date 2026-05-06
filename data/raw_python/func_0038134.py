def bulk_refresh(self):
        """
        Refreshes all refreshable tokens in the queryset.
        Deletes any tokens which fail to refresh.
        Deletes any tokens which are expired and cannot refresh.
        Excludes tokens for which the refresh was incomplete for other reasons.
        """
        session = OAuth2Session(app_settings.ESI_SSO_CLIENT_ID)
        auth = requests.auth.HTTPBasicAuth(app_settings.ESI_SSO_CLIENT_ID, app_settings.ESI_SSO_CLIENT_SECRET)
        incomplete = []
        for model in self.filter(refresh_token__isnull=False):
            try:
                model.refresh(session=session, auth=auth)
                logging.debug("Successfully refreshed {0}".format(repr(model)))
            except TokenError:
                logger.info("Refresh failed for {0}. Deleting.".format(repr(model)))
                model.delete()
            except IncompleteResponseError:
                incomplete.append(model.pk)
        self.filter(refresh_token__isnull=True).get_expired().delete()
        return self.exclude(pk__in=incomplete)
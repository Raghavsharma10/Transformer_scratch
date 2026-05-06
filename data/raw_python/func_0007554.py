def __set_client_detail(self, *args, **kwargs):
        """
        Sets up the ClientDetail node, which is required for all shipping
        related requests.
        """

        client_detail = self.client.factory.create('ClientDetail')
        client_detail.AccountNumber = self.config_obj.account_number
        client_detail.MeterNumber = self.config_obj.meter_number
        client_detail.IntegratorId = self.config_obj.integrator_id
        if hasattr(client_detail, 'Region'):
            client_detail.Region = self.config_obj.express_region_code

        client_language_code = kwargs.get('client_language_code', None)
        client_locale_code = kwargs.get('client_locale_code', None)

        if hasattr(client_detail, 'Localization') and (client_language_code or client_locale_code):
            localization = self.client.factory.create('Localization')

            if client_language_code:
                localization.LanguageCode = client_language_code

            if client_locale_code:
                localization.LocaleCode = client_locale_code

            client_detail.Localization = localization

        self.ClientDetail = client_detail
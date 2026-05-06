def send(self, to, language=None, **data):
        """
        This is the method to be called
        """
        self.data = data
        self.get_context_data()
        if app_settings['SEND_EMAILS']:
            try:
                if language:
                    mail.send(to, template=self.template, context=self.context_data, language=language)
                else:
                    mail.send(to, template=self.template, context=self.context_data)
            except EmailTemplate.DoesNotExist:
                msg = 'Trying to use a non existent email template {0}'.format(self.template)
                LOGGER.error('Trying to use a non existent email template {0}'.format(self.template))
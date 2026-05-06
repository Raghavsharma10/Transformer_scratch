def to_json(self):
        """
        Returns the JSON representation of the locale.
        """

        result = super(Locale, self).to_json()
        result.update({
            'code': self.code,
            'name': self.name,
            'fallbackCode': self.fallback_code,
            'optional': self.optional,
            'contentDeliveryApi': self.content_delivery_api,
            'contentManagementApi': self.content_management_api
        })
        return result
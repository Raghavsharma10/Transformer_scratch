def get_equipment(self, **kwargs):
        """
        Return list environments related with environment vip
        """

        uri = 'api/v3/equipment/'
        uri = self.prepare_url(uri, kwargs)

        return super(ApiEquipment, self).get(uri)
def get_resource_url(self):
        """ Get resource complete url """

        name = self.__class__.resource_name
        url = self.__class__.rest_base_url()

        if self.id is not None:
            return "%s/%s/%s" % (url, name, self.id)

        return "%s/%s" % (url, name)
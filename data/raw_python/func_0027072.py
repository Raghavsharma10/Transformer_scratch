def get_url(self, link):
        """
        URL of service
        """
        view_name = SupportedServices.get_detail_view_for_model(link.service)
        return reverse(view_name, kwargs={'uuid': link.service.uuid.hex}, request=self.context['request'])
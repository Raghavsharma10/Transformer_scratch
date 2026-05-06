def update_links(self, request, admin_site=None):
        """
        Called to update the widget's urls. Tries to find the
        bundle for the model that this foreign key points to and then
        asks it for the urls for adding and listing and sets them on
        this widget instance. The urls are only set if request.user
        has permissions on that url.

        :param request: The request for which this widget is being rendered.
        :param admin_site: If provided, the `admin_site` is used to lookup \
        the bundle that is registered as the primary url for the model \
        that this foreign key points to.
        """
        if admin_site:
            bundle = admin_site.get_bundle_for_model(self.model.to)

            if bundle:
                self._api_link = self._get_bundle_link(bundle, self.view,
                                                       request.user)
                self._add_link = self._get_bundle_link(bundle, self.add_view,
                                                       request.user)
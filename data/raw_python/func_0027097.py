def _get_quick_access_info(self):
        """
        Returns a list of ListLink items to be added to Quick Access tab.
        Contains:
        - links to Organizations, Projects and Users;
        - a link to shared service settings;
        - custom configured links in admin/settings FLUENT_DASHBOARD_QUICK_ACCESS_LINKS attribute;
        """
        quick_access_links = []

        # add custom links
        quick_access_links.extend(settings.FLUENT_DASHBOARD_QUICK_ACCESS_LINKS)

        for model in (structure_models.Project,
                      structure_models.Customer,
                      core_models.User,
                      structure_models.SharedServiceSettings):
            quick_access_links.append(self._get_link_to_model(model))

        return quick_access_links
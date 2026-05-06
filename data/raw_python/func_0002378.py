def refresh_hrefs(self, request):
        """
        Refresh all the cached menu item HREFs in the database.
        """
        for item in treenav.MenuItem.objects.all():
            item.save()  # refreshes the HREF
        self.message_user(request, _('Menu item HREFs refreshed successfully.'))
        info = self.model._meta.app_label, self.model._meta.model_name
        changelist_url = reverse('admin:%s_%s_changelist' % info, current_app=self.admin_site.name)
        return redirect(changelist_url)
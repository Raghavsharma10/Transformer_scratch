def clean_cache(self, request):
        """
        Remove all MenuItems from Cache.
        """
        treenav.delete_cache()
        self.message_user(request, _('Cache menuitem cache cleaned successfully.'))
        info = self.model._meta.app_label, self.model._meta.model_name
        changelist_url = reverse('admin:%s_%s_changelist' % info, current_app=self.admin_site.name)
        return redirect(changelist_url)
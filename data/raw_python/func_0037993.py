def get_apps_menu(self):
        """Temporal code, will change to apps.get_app_configs() for django 1.7

        Generate a initial menu list using the AppsConfig registered
        """
        menu = {}
        for model, model_admin in self.admin_site._registry.items():
            if hasattr(model_admin, 'app_config'):
                if model_admin.app_config.has_menu_permission(obj=self.user):
                    menu.update({
                        'app:' + model_admin.app_config.name: {
                        'title': model_admin.app_config.verbose_name,
                        'menus': model_admin.app_config.init_menu(),
                        'first_icon': model_admin.app_config.icon}})
        return menu
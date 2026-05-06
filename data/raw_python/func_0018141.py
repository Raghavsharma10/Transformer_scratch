def get_menu_item(self):
        """
        Utilised by Wagtail's 'register_menu_item' hook to create a menu
        for this group with a SubMenu linking to listing pages for any
        associated ModelAdmin instances
        """
        if self.modeladmin_instances:
            submenu = SubMenu(self.get_submenu_items())
            return GroupMenuItem(self, self.get_menu_order(), submenu)
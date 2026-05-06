def menu_item(self, sub_assistant, path):
        """
        The function creates a menu item
        and assigns signal like select and button-press-event for
        manipulation with menu_item. sub_assistant and path
        """
        if not sub_assistant[0].icon_path:
            menu_item = self.create_menu_item(sub_assistant[0].fullname)
        else:
            menu_item = self.create_image_menu_item(
                sub_assistant[0].fullname, sub_assistant[0].icon_path
            )
        if sub_assistant[0].description:
            menu_item.set_has_tooltip(True)
            menu_item.connect("query-tooltip",
                              self.parent.tooltip_queries,
                              self.get_formatted_description(sub_assistant[0].description),
            )
        menu_item.connect("select", self.parent.sub_menu_select, path)
        menu_item.connect("button-press-event", self.parent.sub_menu_pressed)
        menu_item.show()
        return menu_item
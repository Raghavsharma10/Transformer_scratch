def build_menu(self, display_type_menu, document_controller, display_panel):
        """Build the dynamic menu for the selected display panel.

        The user accesses this menu by right-clicking on the display panel.

        The basic menu items are to an empty display panel or a browser display panel.

        After that, each display controller factory is given a chance to add to the menu. The display
        controllers (for instance, a scan acquisition controller), may add its own menu items.
        """
        dynamic_live_actions = list()

        def switch_to_display_content(display_panel_type):
            self.switch_to_display_content(document_controller, display_panel, display_panel_type, display_panel.display_item)

        empty_action = display_type_menu.add_menu_item(_("Clear Display Panel"), functools.partial(switch_to_display_content, "empty-display-panel"))
        display_type_menu.add_separator()

        data_item_display_action = display_type_menu.add_menu_item(_("Display Item"), functools.partial(switch_to_display_content, "data-display-panel"))
        thumbnail_browser_action = display_type_menu.add_menu_item(_("Thumbnail Browser"), functools.partial(switch_to_display_content, "thumbnail-browser-display-panel"))
        grid_browser_action = display_type_menu.add_menu_item(_("Grid Browser"), functools.partial(switch_to_display_content, "browser-display-panel"))
        display_type_menu.add_separator()

        display_panel_type = display_panel.display_panel_type

        empty_action.checked = display_panel_type == "empty" and display_panel.display_panel_controller is None
        data_item_display_action.checked = display_panel_type == "data_item"
        thumbnail_browser_action.checked = display_panel_type == "horizontal"
        grid_browser_action.checked = display_panel_type == "grid"

        dynamic_live_actions.append(empty_action)
        dynamic_live_actions.append(data_item_display_action)
        dynamic_live_actions.append(thumbnail_browser_action)
        dynamic_live_actions.append(grid_browser_action)

        for factory in self.__display_controller_factories.values():
            dynamic_live_actions.extend(factory.build_menu(display_type_menu, display_panel))

        return dynamic_live_actions
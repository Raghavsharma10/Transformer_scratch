def add_submenu(self, grid_lang, ass, row, column):
        """
        The function is used for creating button with menu and submenu.
        Also signal on tooltip and signal on clicked are specified
        Button is add to the Gtk.Grid
        """
        text = "Available subassistants:\n"
        # Generate menus
        path = []
        (menu, text) = self.generate_menu(ass, text, path=path)
        menu.show_all()
        if ass[0].description:
            description = self.get_formatted_description(ass[0].description) + "\n\n"
        else:
            description = ""
        description += text.replace('|', '\n')
        image_name = ass[0].icon_path
        lbl_text = "<b>" + ass[0].fullname + "</b>"
        if not image_name:
            btn = self.button_with_label(lbl_text)
        else:
            btn = self.button_with_image(lbl_text, image=image_name)
        btn.set_has_tooltip(True)
        btn.connect("query-tooltip",
                    self.parent.tooltip_queries,
                    description
        )
        btn.connect_object("event", self.parent.btn_press_event, menu)
        if row == 0 and column == 0:
            grid_lang.add(btn)
        else:
            grid_lang.attach(btn, column, row, 1, 1)
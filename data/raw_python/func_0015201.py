def add_button(self, grid_lang, ass, row, column):
        """
        The function is used for creating button with all features
        like signal on tooltip and signal on clicked
        The function does not have any menu.
        Button is add to the Gtk.Grid on specific row and column
        """
        #print "gui_helper add_button"
        image_name = ass[0].icon_path
        label = "<b>" + ass[0].fullname + "</b>"
        if not image_name:
            btn = self.button_with_label(label)
        else:
            btn = self.button_with_image(label, image=ass[0].icon_path)
        #print "Dependencies button",ass[0]._dependencies
        if ass[0].description:
            btn.set_has_tooltip(True)
            btn.connect("query-tooltip",
                        self.parent.tooltip_queries,
                        self.get_formatted_description(ass[0].description)
            )
        btn.connect("clicked", self.parent.btn_clicked, ass[0].name)
        if row == 0 and column == 0:
            grid_lang.add(btn)
        else:
            grid_lang.attach(btn, column, row, 1, 1)
        return btn
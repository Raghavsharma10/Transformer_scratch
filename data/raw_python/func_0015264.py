def _create_notebook_page(self, assistant):
        """
        This function is used for create tab page for notebook.
        Input arguments are:
        assistant - used for collecting all info about assistants and subassistants
        """
        #frame = self._create_frame()
        grid_lang = self.gui_helper.create_gtk_grid()
        scrolled_window = self.gui_helper.create_scrolled_window(grid_lang)
        row = 0
        column = 0
        scrolled_window.main_assistant, sub_as = assistant.get_subassistant_tree()
        for ass in sorted(sub_as, key=lambda x: x[0].fullname.lower()):
            if column > 2:
                row += 1
                column = 0
            if not ass[1]:
                # If assistant has not any subassistant then create only button
                self.gui_helper.add_button(grid_lang, ass, row, column)
            else:
                # If assistant has more subassistants then create button with menu
                self.gui_helper.add_submenu(grid_lang, ass, row, column)
            column += 1

        # Install More Assistants button
        if column > 2:
            row += 1
            column = 0
        self.gui_helper.add_install_button(grid_lang, row, column)
        column += 1

        if row == 0 and len(sub_as) < 3:
            while column < 3:
                btn = self.gui_helper.create_button(style=Gtk.ReliefStyle.NONE)
                btn.set_sensitive(False)
                btn.hide()
                grid_lang.attach(btn, column, row, 1, 1)
                column += 1
        return scrolled_window
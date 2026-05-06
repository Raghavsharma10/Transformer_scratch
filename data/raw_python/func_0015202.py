def add_install_button(self, grid_lang, row, column):
        """
        Add button that opens the window for installing more assistants
        """
        btn = self.button_with_label('<b>Install more...</b>')
        if row == 0 and column == 0:
            grid_lang.add(btn)
        else:
            grid_lang.attach(btn, column, row, 1, 1)
        btn.connect("clicked", self.parent.install_btn_clicked)
        return btn
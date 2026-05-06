def __replace_displayed_display_item(self, display_panel, display_item, d=None) -> Undo.UndoableCommand:
        """ Used in drag/drop support. """
        self.document_controller.replaced_display_panel_content = display_panel.save_contents()
        command = DisplayPanel.ReplaceDisplayPanelCommand(self)
        if display_item:
            display_panel.set_display_panel_display_item(display_item, detect_controller=True)
        elif d is not None:
            display_panel.change_display_panel_content(d)
        display_panel.request_focus()
        self.__sync_layout()
        return command
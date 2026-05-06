def selected_display_item(self) -> typing.Optional[DisplayItem.DisplayItem]:
        """Return the selected display item.

        The selected display is the display ite that has keyboard focus in the data panel or a display panel.
        """
        # first check for the [focused] data browser
        display_item = self.focused_display_item
        if not display_item:
            selected_display_panel = self.selected_display_panel
            display_item = selected_display_panel.display_item if selected_display_panel else None
        return display_item
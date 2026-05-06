def all_display_panels(self) -> typing.List[DisplayPanel]:
        """Return the list of display panels currently visible.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        return [DisplayPanel(display_panel) for display_panel in self.__document_controller.workspace_controller.display_panels]
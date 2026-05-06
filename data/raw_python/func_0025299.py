def get_display_panel_by_id(self, identifier: str) -> DisplayPanel:
        """Return display panel with the identifier.

        .. versionadded:: 1.0

        Status: Provisional
        Scriptable: Yes
        """
        display_panel = next(
            (display_panel for display_panel in self.__document_controller.workspace_controller.display_panels if
            display_panel.identifier.lower() == identifier.lower()), None)
        return DisplayPanel(display_panel) if display_panel else None
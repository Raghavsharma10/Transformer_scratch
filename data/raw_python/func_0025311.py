def create_panel(self, panel_delegate):
        """Create a utility panel that can be attached to a window.

        .. versionadded:: 1.0

        Scriptable: No

         The panel_delegate should respond to the following:
            (property, read-only) panel_id
            (property, read-only) panel_name
            (property, read-only) panel_positions (a list from "top", "bottom", "left", "right", "all")
            (property, read-only) panel_position (from "top", "bottom", "left", "right", "none")
            (method, required) create_panel_widget(ui), returns a widget
            (method, optional) close()
        """

        panel_id = panel_delegate.panel_id
        panel_name = panel_delegate.panel_name
        panel_positions = getattr(panel_delegate, "panel_positions", ["left", "right"])
        panel_position = getattr(panel_delegate, "panel_position", "none")
        properties = getattr(panel_delegate, "panel_properties", None)

        workspace_manager = Workspace.WorkspaceManager()

        def create_facade_panel(document_controller, panel_id, properties):
            panel = Panel(document_controller, panel_id, properties)
            ui = UserInterface(self.__ui_version, document_controller.ui)
            document_controller = DocumentWindow(document_controller)
            panel.widget = panel_delegate.create_panel_widget(ui, document_controller)._widget
            return panel

        class PanelReference:

            def __init__(self):
                self.__panel_delegate = panel_delegate
                workspace_manager.register_panel(create_facade_panel, panel_id, panel_name, panel_positions, panel_position, properties)

            def __del__(self):
                self.close()

            def close(self):
                if self.__panel_delegate:
                    panel_delegate_close_fn = getattr(self.__panel_delegate, "close", None)
                    if panel_delegate_close_fn:
                       panel_delegate_close_fn()
                    workspace_manager.unregister_panel(panel_id)
                    self.__panel_delegate = None

        return PanelReference()
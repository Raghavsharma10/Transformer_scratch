def display_data_item(self, data_item: DataItem, source_display_panel=None, source_data_item=None):
        """Display a new data item and gives it keyboard focus. Uses existing display if it is already displayed.

        .. versionadded:: 1.0

        Status: Provisional
        Scriptable: Yes
        """
        for display_panel in self.__document_controller.workspace_controller.display_panels:
            if display_panel.data_item == data_item._data_item:
                display_panel.request_focus()
                return DisplayPanel(display_panel)
        result_display_panel = self.__document_controller.next_result_display_panel()
        if result_display_panel:
            display_item = self.__document_controller.document_model.get_display_item_for_data_item(data_item._data_item)
            result_display_panel.set_display_panel_display_item(display_item)
            result_display_panel.request_focus()
            return DisplayPanel(result_display_panel)
        return None
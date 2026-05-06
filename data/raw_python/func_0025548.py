def __create_list_item_widget(self, ui, calibration_observable):
        """Called when an item (calibration_observable) is inserted into the list widget. Returns a widget."""
        calibration_row = make_calibration_row_widget(ui, calibration_observable)
        column = ui.create_column_widget()
        column.add_spacing(4)
        column.add(calibration_row)
        return column
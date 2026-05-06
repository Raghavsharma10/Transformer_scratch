def make_calibration_row_widget(ui, calibration_observable, label: str=None):
    """Called when an item (calibration_observable) is inserted into the list widget. Returns a widget."""
    calibration_row = ui.create_row_widget()
    row_label = ui.create_label_widget(label, properties={"width": 60})
    row_label.widget_id = "label"
    offset_field = ui.create_line_edit_widget(properties={"width": 60})
    offset_field.widget_id = "offset"
    scale_field = ui.create_line_edit_widget(properties={"width": 60})
    scale_field.widget_id = "scale"
    units_field = ui.create_line_edit_widget(properties={"width": 60})
    units_field.widget_id = "units"
    float_point_4_converter = Converter.FloatToStringConverter(format="{0:.4f}")
    offset_field.bind_text(Binding.PropertyBinding(calibration_observable, "offset", converter=float_point_4_converter))
    scale_field.bind_text(Binding.PropertyBinding(calibration_observable, "scale", converter=float_point_4_converter))
    units_field.bind_text(Binding.PropertyBinding(calibration_observable, "units"))
    # notice the binding of calibration_index below.
    calibration_row.add(row_label)
    calibration_row.add_spacing(12)
    calibration_row.add(offset_field)
    calibration_row.add_spacing(12)
    calibration_row.add(scale_field)
    calibration_row.add_spacing(12)
    calibration_row.add(units_field)
    calibration_row.add_stretch()
    return calibration_row
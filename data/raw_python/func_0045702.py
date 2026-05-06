def set_cell_value(cell, value):
    """
    Convenience method for setting the value of an openpyxl cell

    This is necessary since the value property changed from internal_value
    to value between version 1.* and 2.*.
    """
    if OPENPYXL_MAJOR_VERSION > 1:
        cell.value = value
    else:
        cell.internal_value = value
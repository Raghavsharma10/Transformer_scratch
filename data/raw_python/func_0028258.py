def create_excel_workbook(data, result_info_key, identifier_keys):
    """Calls the analytics_data_excel module to create the Workbook"""
    workbook = analytics_data_excel.get_excel_workbook(data, result_info_key, identifier_keys)
    adjust_column_width_workbook(workbook)
    return workbook
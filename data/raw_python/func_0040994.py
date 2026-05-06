def post_process(table, post_processors):
    """Applies the list of post processing methods if any"""
    table_result = table
    for processor in post_processors:
        table_result = processor(table_result)
    return table_result
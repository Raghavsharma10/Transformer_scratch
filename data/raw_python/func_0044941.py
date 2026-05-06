def export_data(target_path):
    """
    Exports the data of an application - media files plus database,
    :param: target_path:
    :return: a zip archive
    """
    tasks.export_data_dir(target_path)
    tasks.export_database(target_path)
    tasks.export_context(target_path)
    return target_path
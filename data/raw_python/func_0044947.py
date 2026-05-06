def import_context(target_zip):
    """
    Overwrite old context.json, use context.json from target_zip
    :param target_zip:
    :return:
    """
    context_path = tasks.get_context_path()
    with zipfile.ZipFile(target_zip) as unzipped_data:
        with open(context_path, 'w') as context:
            context.write(unzipped_data.read('context.json'))
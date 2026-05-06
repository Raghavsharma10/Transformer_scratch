def import_data(target_zip):
    """
    Import data from given zip-arc, this means database + __data__
    :param target_zip:
    :param backup_zip_path:
    :return:
    """
    from django_productline.context import PRODUCT_CONTEXT
    tasks.import_data_dir(target_zip)
    # product context is not reloaded if context file is changed
    tasks.import_database(target_zip, PRODUCT_CONTEXT.DB_NAME, PRODUCT_CONTEXT.DB_USER)
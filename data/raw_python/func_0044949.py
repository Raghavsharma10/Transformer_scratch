def create_data_dir():
    """
    Creates the DATA_DIR.
    :return:
    """
    from django_productline.context import PRODUCT_CONTEXT
    if not os.path.exists(PRODUCT_CONTEXT.DATA_DIR):
        os.mkdir(PRODUCT_CONTEXT.DATA_DIR)
        print('*** Created DATA_DIR in %s' % PRODUCT_CONTEXT.DATA_DIR)
    else:
        print('...DATA_DIR already exists.')
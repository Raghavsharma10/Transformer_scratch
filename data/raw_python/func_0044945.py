def mv_data_dir(target):
    """
    Move data_dir to {target} location, refineable in case data_dir is a mounted volume or object storage and needs special treatments
    :return:
    """
    from django_productline.context import PRODUCT_CONTEXT
    os.rename(PRODUCT_CONTEXT.DATA_DIR, target)
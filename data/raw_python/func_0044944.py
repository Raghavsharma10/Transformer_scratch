def import_data_dir(target_zip):
    """
    Imports the data specified by param <target_zip>. Renames the data dir if it already exists and
    unpacks the zip sub dir __data__ directly within the current active product.
    :param target_zip: string path to the zip file.
    """
    from django_productline.context import PRODUCT_CONTEXT

    new_data_dir = '{data_dir}_before_import_{ts}'.format(
        data_dir=PRODUCT_CONTEXT.DATA_DIR,
        ts=datetime.datetime.now().strftime("%Y-%m-%d.%H:%M:%S:%s")
    )

    if os.path.exists(PRODUCT_CONTEXT.DATA_DIR):
        # rename an existing data dir if it exists
        tasks.mv_data_dir(new_data_dir)

    z = zipfile.ZipFile(target_zip)

    def filter_func(x):
        return x.startswith('__data__/')

    z.extractall(os.path.dirname(PRODUCT_CONTEXT.DATA_DIR), filter(filter_func, z.namelist()))
def export_data_dir(target_path):
    """
    Exports the media files of the application and bundles a zip archive
    :return: the target path of the zip archive
    """
    from django_productline import utils
    from django.conf import settings

    utils.zipdir(settings.PRODUCT_CONTEXT.DATA_DIR, target_path, wrapdir='__data__')
    print('... wrote {target_path}'.format(target_path=target_path))
    return target_path
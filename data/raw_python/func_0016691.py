def image_post_delete_handler(sender, instance, **kwargs):
    """
    Makes sure that a an image is also deleted from the media directory.

    This should prevent a load of "dead" image files on disc.

    """
    for f in glob.glob('{}/{}*'.format(instance.image.storage.location,
                                       instance.image.name)):
        if not os.path.isdir(f):
            instance.image.storage.delete(f)
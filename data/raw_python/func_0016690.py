def get_image_file_path(instance, filename):
    """Returns a unique filename for images."""
    ext = filename.split('.')[-1]
    filename = '%s.%s' % (uuid.uuid4(), ext)
    return os.path.join(
        'user_media', str(instance.user.pk), 'images', filename)
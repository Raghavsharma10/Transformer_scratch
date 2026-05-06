def get_imagestreamtag_from_image(image):
    """
    return ImageStreamTag, give a FROM value

    :param image: str, the FROM value from the Dockerfile
    :return: str, ImageStreamTag
    """
    ret = image

    # Remove the registry part
    ret = strip_registry_from_image(image)

    # ImageStream names cannot contain '/'
    ret = ret.replace('/', '-')

    # If there is no ':' suffix value, add one
    if ret.find(':') == -1:
        ret += ":latest"

    return ret
def _replace_image(image_url, image_tag, ebook_folder,
                   image_name=None):
    """
    Replaces the src of an image to link to the local copy in the images folder of the ebook. Tightly coupled with bs4
        package.

    Args:
        image_url (str): The url of the image.
        image_tag (bs4.element.Tag): The bs4 tag containing the image.
        ebook_folder (str): The directory where the ebook files are being saved. This must contain a subdirectory
            called "images".
        image_name (Option[str]): The short name to save the image as. Should not contain a directory or an extension.
    """
    try:
        assert isinstance(image_tag, bs4.element.Tag)
    except AssertionError:
        raise TypeError("image_tag cannot be of type " + str(type(image_tag)))
    if image_name is None:
        image_name = str(uuid.uuid4())
    try:
        image_full_path = os.path.join(ebook_folder, 'images')
        assert os.path.exists(image_full_path)
        image_extension = save_image(image_url, image_full_path,
                                     image_name)
        image_tag['src'] = 'images' + '/' + image_name + '.' + image_extension
    except ImageErrorException:
        image_tag.decompose()
    except AssertionError:
        raise ValueError('%s doesn\'t exist or doesn\'t contain a subdirectory images' % ebook_folder)
    except TypeError:
        image_tag.decompose()
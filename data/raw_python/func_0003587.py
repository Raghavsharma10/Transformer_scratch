def save_image(image_url, image_directory, image_name):
    """
    Saves an online image from image_url to image_directory with the name image_name.
    Returns the extension of the image saved, which is determined dynamically.

    Args:
        image_url (str): The url of the image.
        image_directory (str): The directory to save the image in.
        image_name (str): The file name to save the image as.

    Raises:
        ImageErrorException: Raised if unable to save the image at image_url
    """
    image_type = get_image_type(image_url)
    if image_type is None:
        raise ImageErrorException(image_url)
    full_image_file_name = os.path.join(image_directory, image_name + '.' + image_type)

    # If the image is present on the local filesystem just copy it
    if os.path.exists(image_url):
        shutil.copy(image_url, full_image_file_name)
        return image_type

    try:
        # urllib.urlretrieve(image_url, full_image_file_name)
        with open(full_image_file_name, 'wb') as f:
            user_agent = r'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0'
            request_headers = {'User-Agent': user_agent}
            requests_object = requests.get(image_url, headers=request_headers)
            try:
                content = requests_object.content
                # Check for empty response
                f.write(content)
            except AttributeError:
                raise ImageErrorException(image_url)
    except IOError:
        raise ImageErrorException(image_url)
    return image_type
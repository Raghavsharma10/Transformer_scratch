def get_one_aminame(inst_img_id):
    """Get Image_Name for the image_id specified.

    Args:
        inst_img_id (str): image_id to get name value from.
    Returns:
        aminame (str): name of the image.

    """
    try:
        aminame = EC2R.Image(inst_img_id).name
    except AttributeError:
        aminame = "Unknown"
    return aminame
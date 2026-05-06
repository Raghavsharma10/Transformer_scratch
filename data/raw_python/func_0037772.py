def image(title, desc, image_name, group=None, height=None):
    """
    Builds an image element.  Image elements are primarily created
    and then wrapped into an image gallery element.  This is not required
    behavior, however and it's independent usage should be allowed depending
    on the behavior required.

    The Javascript will search for the `image_name` in the component's
    `imgs` directory when rendering.  For example, all verification images
    are output to `vv_xxxx-xx-xx/verification/imgs` and then the verification
    case's output page will search for `image_name` within that directory.

    Args:
        title: The title to display
        desc: A description of the image or plot
        image_name: The filename of the image
        group: (optional) Title of lightbox group to join
        height: (optional) Height of image thumbnail to draw

    Returns:
        A dictionary with the metadata specifying that it is to be
        rendered as an image element
    """
    ie = {
          'Type': 'Image',
          'Title': title,
          'Description': desc,
          'Plot File': image_name,
          }
    if group:
        ie['Group'] = group
    if height:
        ie['Height'] = height
    return ie
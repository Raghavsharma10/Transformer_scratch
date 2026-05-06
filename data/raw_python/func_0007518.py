def view_for_image_named(image_name):
    """Create an ImageView for the given image."""

    image = resource.get_image(image_name)

    if not image:
        return None

    return ImageView(pygame.Rect(0, 0, 0, 0), image)
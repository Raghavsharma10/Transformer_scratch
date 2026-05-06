def _check_img_lib():
    """Utility to search for imageio or PIL"""
    # Import imageio or PIL
    imageio = PIL = None
    try:
        import imageio
    except ImportError:
        try:
            import PIL.Image
        except ImportError:
            pass
    return imageio, PIL
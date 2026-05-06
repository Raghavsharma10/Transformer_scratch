def raster_to_shape(raster):
    """Take a raster and return a polygon representing the outer edge."""
    left = raster.bounds.left
    right = raster.bounds.right
    top = raster.bounds.top
    bottom = raster.bounds.bottom

    top_left = (left, top)
    top_right = (right, top)
    bottom_left = (left, bottom)
    bottom_right = (right, bottom)

    return Polygon((
        top_left, top_right, bottom_right, bottom_left, top_left,
    ))
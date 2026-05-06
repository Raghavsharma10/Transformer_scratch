def calculate_origin_and_size(canvas_size, data_shape, image_canvas_mode, image_zoom, image_position) -> typing.Tuple[typing.Any, typing.Any]:
    """Calculate origin and size for canvas size, data shape, and image display parameters."""
    if data_shape is None:
        return None, None
    if image_canvas_mode == "fill":
        data_shape = data_shape
        scale_h = float(data_shape[1]) / canvas_size[1]
        scale_v = float(data_shape[0]) / canvas_size[0]
        if scale_v < scale_h:
            image_canvas_size = (canvas_size[0], canvas_size[0] * data_shape[1] / data_shape[0])
        else:
            image_canvas_size = (canvas_size[1] * data_shape[0] / data_shape[1], canvas_size[1])
        image_canvas_origin = (canvas_size[0] * 0.5 - image_canvas_size[0] * 0.5, canvas_size[1] * 0.5 - image_canvas_size[1] * 0.5)
    elif image_canvas_mode == "fit":
        image_canvas_size = canvas_size
        image_canvas_origin = (0, 0)
    elif image_canvas_mode == "1:1":
        image_canvas_size = data_shape
        image_canvas_origin = (canvas_size[0] * 0.5 - image_canvas_size[0] * 0.5, canvas_size[1] * 0.5 - image_canvas_size[1] * 0.5)
    elif image_canvas_mode == "2:1":
        image_canvas_size = (data_shape[0] * 0.5, data_shape[1] * 0.5)
        image_canvas_origin = (canvas_size[0] * 0.5 - image_canvas_size[0] * 0.5, canvas_size[1] * 0.5 - image_canvas_size[1] * 0.5)
    else:
        image_canvas_size = (canvas_size[0] * image_zoom, canvas_size[1] * image_zoom)
        canvas_rect = Geometry.fit_to_size(((0, 0), image_canvas_size), data_shape)
        image_canvas_origin_y = (canvas_size[0] * 0.5) - image_position[0] * canvas_rect[1][0] - canvas_rect[0][0]
        image_canvas_origin_x = (canvas_size[1] * 0.5) - image_position[1] * canvas_rect[1][1] - canvas_rect[0][1]
        image_canvas_origin = (image_canvas_origin_y, image_canvas_origin_x)
    return image_canvas_origin, image_canvas_size
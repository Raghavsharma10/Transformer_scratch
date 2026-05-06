def array_split(
    ary,
    indices_or_sections=None,
    axis=None,
    tile_shape=None,
    max_tile_bytes=None,
    max_tile_shape=None,
    sub_tile_shape=None,
    halo=None
):
    "To be replaced."
    return [
        ary[slyce]
        for slyce in
        shape_split(
            array_shape=ary.shape,
            indices_or_sections=indices_or_sections,
            axis=axis,
            array_start=None,
            array_itemsize=ary.itemsize,
            tile_shape=tile_shape,
            max_tile_bytes=max_tile_bytes,
            max_tile_shape=max_tile_shape,
            sub_tile_shape=sub_tile_shape,
            halo=halo,
            tile_bounds_policy=ARRAY_BOUNDS
        ).flatten()
    ]
def sprite_filepath_build(sprite_type, sprite_id, **kwargs):
    """returns the filepath of the sprite *relative to SPRITE_CACHE*"""

    options = parse_sprite_options(sprite_type, **kwargs)

    filename = '.'.join([str(sprite_id), SPRITE_EXT])
    filepath = os.path.join(sprite_type, *options, filename)

    return filepath
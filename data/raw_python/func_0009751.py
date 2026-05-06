def turtle_to_texture(turtle_program, turn_amount=DEFAULT_TURN,
                      initial_angle=DEFAULT_INITIAL_ANGLE, resolution=1):
    """Makes a texture from a turtle program.

    Args:
        turtle_program (str): a string representing the turtle program; see the
            docstring of `branching_turtle_generator` for more details
        turn_amount (float): amount to turn in degrees
        initial_angle (float): initial orientation of the turtle
        resolution (int): if provided, interpolation amount for visible lines

    Returns:
        texture: A texture.
    """
    generator = branching_turtle_generator(
        turtle_program, turn_amount, initial_angle, resolution)
    return texture_from_generator(generator)
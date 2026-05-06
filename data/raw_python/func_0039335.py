def shapesides(inputtocheck, inputtype='shape'):
    """
    Get the sides of a shape.

    inputtocheck:
    The amount of sides or the shape to be checked,
    depending on the value of inputtype.

    inputtype:
    The type of input provided.
    Can be: 'shape', 'sides'.
    """

    # Define the array of sides to a shape
    shapestosides = {
        'triangle': 3,
        'square': 4,
        'pentagon': 5,
        'hexagon': 6,
        'heptagon': 7,
        'octagon': 8,
        'nonagon': 9,
        'decagon': 10,
        'hendecagon': 11,
        'dodecagon': 12,
        'triskaidecagon': 13,
        'tetrakaidecagon': 14,
        'pentadecagon': 15,
        'hexakaidecagon': 16,
        'heptadecagon': 17,
        'octakaidecagon': 18,
        'enneadecagon': 19,
        'icosagon': 20,
        'triacontagon': 30,
        'tetracontagon': 40,
        'pentacontagon': 50,
        'hexacontagon': 60,
        'heptacontagon': 70,
        'octacontagon': 80,
        'enneacontagon': 90,
        'hectagon': 100,
        'chiliagon': 1000,
        'myriagon': 10000,
        'megagon': 1000000,
        'googolgon': pow(10, 100),
        'ngon': 'n'
    }

    # Define an array with the flipped version of the sides to a shape
    sidestoshapes = dictflip(shapestosides)

    # If the lowercase version of the input type is 'shape'
    if inputtype.lower() == 'shape':
        # If the lowercase version of the shape is in the array
        if inputtocheck.lower() in shapestosides:
            # Return the corresponding sides
            return shapestosides[inputtocheck.lower()]

        # Return 'n'
        return shapestosides['n']

    if inputtype.lower() == 'sides':
        # If the lowercase version of the shape is in the array
        if inputtocheck.lower() in sidestoshapes:
            # Return the corresponding sides
            return sidestoshapes[inputtocheck.lower()]

        # Return 'ngon'
        return sidestoshapes['ngon']

    # Raise a warning
    raise ValueError("Invalid input type.")
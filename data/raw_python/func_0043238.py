def qr2scad(stream):
    """Convert black pixels to OpenSCAD cubes."""

    img = Image.open(stream)

    # Convert to black and white 8-bit
    if img.mode != 'L':
        img = img.convert('L')

    # Invert color to get the right bounding box
    img = ImageOps.invert(img)

    bbox = img.getbbox()

    # Crop to only contain contents within the PDPs
    img = img.crop(bbox)

    width, height = img.size

    assert width == height,\
        'The QR code should be a square, but we found it to be %(w)sx%(h)s' % {
            'w': width,
            'h': height
        }

    qr_side = width

    # QR code superpixel size
    qr_pixel_size = (list(img.getdata()).index(0) / PDP_SIDE)

    # Get the resize factor from the PDP size
    new_size = qr_side / qr_pixel_size

    # Set a more reasonable size
    img = img.resize((new_size, new_size))
    qr_side = new_size

    img_matrix = img.load()

    result = 'module _qr_code_dot() {\n'
    result += '    cube([%(block_side)s, %(block_side)s, 1]);\n' % {
        'block_side': BLOCK_SIDE
    }
    result += '}\n'

    result += 'module qr_code() {\n'
    for row in range(qr_side):
        for column in range(qr_side):
            if img_matrix[column, row] != 0:
                result += '    translate([%(x)s, %(y)s, 0])' % {
                    'x': BLOCK_SIZE * column - qr_side / 2,
                    'y': -BLOCK_SIZE * row + qr_side / 2
                }
                result += ' _qr_code_dot();\n'
    result += '}\n'
    result += 'qr_code_size = %d;' % (qr_side)

    return result
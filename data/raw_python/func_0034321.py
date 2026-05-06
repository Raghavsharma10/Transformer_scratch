def add_cell(preso, pos, width, height, padding=1, top_margin=4, left_margin=2):
    """ Add a text frame to current slide """
    available_width = SLIDE_WIDTH
    available_width -= left_margin * 2
    available_width -= padding * (width - 1)
    column_width = available_width / width
    avail_height = SLIDE_HEIGHT
    avail_height -= top_margin
    avail_height -= padding * (height - 1)
    column_height = avail_height / height

    col_pos = int((pos - 1) % width)
    row_pos = int((pos - 1) / width)

    w = "{}cm".format(column_width)
    h = "{}cm".format(column_height)
    x = "{}cm".format(left_margin + (col_pos * column_width + (col_pos) * padding))
    y = "{}cm".format(top_margin + (row_pos * column_height + (row_pos) * padding))
    attr = {
        "presentation:class": "outline",
        "presentation:style-name": "Default-outline1",
        "svg:width": w,
        "svg:height": h,
        "svg:x": x,
        "svg:y": y,
    }
    preso.slides[-1].add_text_frame(attr)
    preso.slides[-1].grid_w_h_x_y = (w, h, x, y)
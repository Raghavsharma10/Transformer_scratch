def line(surf, start, end, color=BLACK, width=1, style=FLAT):
    """Draws an antialiased line on the surface."""

    width = round(width, 1)
    if width == 1:
        # return pygame.draw.aaline(surf, color, start, end)
        return gfxdraw.line(surf, *start, *end, color)

    start = V2(*start)
    end = V2(*end)

    line_vector = end - start
    half_side = line_vector.normnorm() * width / 2

    point1 = start + half_side
    point2 = start - half_side
    point3 = end - half_side
    point4 = end + half_side

    # noinspection PyUnresolvedReferences
    liste = [
        (point1.x, point1.y),
        (point2.x, point2.y),
        (point3.x, point3.y),
        (point4.x, point4.y)
    ]

    rect = polygon(surf, liste, color)

    if style == ROUNDED:
        _ = circle(surf, start, width / 2, color)
        rect = merge_rects(rect, _)
        _ = circle(surf, end, width / 2, color)
        rect = merge_rects(rect, _)

    return rect
def ring(surf, xy, r, width, color):
    """Draws a ring"""

    r2 = r - width

    x0, y0 = xy
    x = r2
    y = 0
    err = 0

    # collect points of the inner circle
    right = {}
    while x >= y:
        right[x] = y
        right[y] = x
        right[-x] = y
        right[-y] = x

        y += 1
        if err <= 0:
            err += 2 * y + 1
        if err > 0:
            x -= 1
            err -= 2 * x + 1

    def h_fill_the_circle(surf, color, x, y, right):
        if -r2 <= y <= r2:
            pygame.draw.line(surf, color, (x0 + right[y], y0 + y), (x0 + x, y0 + y))
            pygame.draw.line(surf, color, (x0 - right[y], y0 + y), (x0 - x, y0 + y))
        else:
            pygame.draw.line(surf, color, (x0 - x, y0 + y), (x0 + x, y0 + y))

    x = r
    y = 0
    err = 0

    while x >= y:

        h_fill_the_circle(surf, color, x, y, right)
        h_fill_the_circle(surf, color, x, -y, right)
        h_fill_the_circle(surf, color, y, x, right)
        h_fill_the_circle(surf, color, y, -x, right)

        y += 1
        if err < 0:
            err += 2 * y + 1
        if err >= 0:
            x -= 1
            err -= 2 * x + 1

    gfxdraw.aacircle(surf, x0, y0, r, color)
    gfxdraw.aacircle(surf, x0, y0, r2, color)
def circle(surf, xy, r, color=BLACK):
    """Draw an antialiased filled circle on the given surface"""

    x, y = xy

    x = round(x)
    y = round(y)
    r = round(r)

    gfxdraw.filled_circle(surf, x, y, r, color)
    gfxdraw.aacircle(surf, x, y, r, color)

    r += 1
    return pygame.Rect(x - r, y - r, 2 * r, 2 * r)
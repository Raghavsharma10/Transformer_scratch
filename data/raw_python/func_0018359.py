def roundrect(surface, rect, color, rounding=5, unit=PIXEL):
    """
    Draw an antialiased round rectangle on the surface.

    surface : destination
    rect    : rectangle
    color   : rgb or rgba
    radius  : 0 <= radius <= 1
    :source: http://pygame.org/project-AAfilledRoundedRect-2349-.html
    """

    if unit == PERCENT:
        rounding = int(min(rect.size) / 2 * rounding / 100)

    rect = pygame.Rect(rect)
    color = pygame.Color(*color)
    alpha = color.a
    color.a = 0
    pos = rect.topleft
    rect.topleft = 0, 0
    rectangle = pygame.Surface(rect.size, SRCALPHA)

    circle = pygame.Surface([min(rect.size) * 3] * 2, SRCALPHA)
    pygame.draw.ellipse(circle, (0, 0, 0), circle.get_rect(), 0)
    circle = pygame.transform.smoothscale(circle, (rounding, rounding))

    rounding = rectangle.blit(circle, (0, 0))
    rounding.bottomright = rect.bottomright
    rectangle.blit(circle, rounding)
    rounding.topright = rect.topright
    rectangle.blit(circle, rounding)
    rounding.bottomleft = rect.bottomleft
    rectangle.blit(circle, rounding)

    rectangle.fill((0, 0, 0), rect.inflate(-rounding.w, 0))
    rectangle.fill((0, 0, 0), rect.inflate(0, -rounding.h))

    rectangle.fill(color, special_flags=BLEND_RGBA_MAX)
    rectangle.fill((255, 255, 255, alpha), special_flags=BLEND_RGBA_MIN)

    return surface.blit(rectangle, pos)
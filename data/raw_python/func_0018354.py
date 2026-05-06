def merge_rects(rect1, rect2):
    """Return the smallest rect containning two rects"""
    r = pygame.Rect(rect1)
    t = pygame.Rect(rect2)

    right = max(r.right, t.right)
    bot = max(r.bottom, t.bottom)
    x = min(t.x, r.x)
    y = min(t.y, r.y)

    return pygame.Rect(x, y, right - x, bot - y)
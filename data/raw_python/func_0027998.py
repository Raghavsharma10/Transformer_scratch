def parse_series(s):
    """
    Parses things like '1n+2', or 'an+b' generally, returning (a, b)
    """
    if isinstance(s, Element):
        s = s._format_element()
    if not s or s == '*':
        # Happens when there's nothing, which the CSS parser thinks of as *
        return (0, 0)
    if isinstance(s, int):
        # Happens when you just get a number
        return (0, s)
    if s == 'odd':
        return (2, 1)
    elif s == 'even':
        return (2, 0)
    elif s == 'n':
        return (1, 0)
    if 'n' not in s:
        # Just a b
        return (0, int(s))
    a, b = s.split('n', 1)
    if not a:
        a = 1
    elif a == '-' or a == '+':
        a = int(a+'1')
    else:
        a = int(a)
    if not b:
        b = 0
    elif b == '-' or b == '+':
        b = int(b+'1')
    else:
        b = int(b)
    return (a, b)
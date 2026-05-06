def t_to_min(x):
    """
    Convert XML 'xs: duration type' to decimal minutes, e.g.:
    t_to_min('PT1H2M30S') == 62.5
    """
    g = re.match('PT(?:(.*)H)?(?:(.*)M)?(?:(.*)S)?', x).groups()
    return sum(0 if g[i] is None else float(g[i]) * 60. ** (1 - i)
               for i in range(3))
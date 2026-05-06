def rgb_to_short(rgb, mapping):
    """Find the closest xterm-256 approximation to the given RGB value."""
    # Thanks to Micah Elliott (http://MicahElliott.com) for colortrans.py
    rgb = rgb.lstrip('#') if rgb.startswith('#') else rgb
    incs = (0x00, 0x5f, 0x87, 0xaf, 0xd7, 0xff)
    # Break 6-char RGB code into 3 integer vals.
    parts = [int(h, 16) for h in re.split(r'(..)(..)(..)', rgb)[1:4]]
    res = []
    for part in parts:
        i = 0
        while i < len(incs)-1:
            s, b = incs[i], incs[i+1]  # smaller, bigger
            if s <= part <= b:
                s1 = abs(s - part)
                b1 = abs(b - part)
                if s1 < b1: closest = s
                else: closest = b
                res.append(closest)
                break
            i += 1
    res = ''.join([ ('%02.x' % i) for i in res ])
    equiv = mapping[res]
    return equiv, res
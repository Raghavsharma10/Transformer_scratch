def ipv6_prefix_to_mask(prefix):
    """
    ipv6 cidr prefix to net mask

    :param prefix: cidr prefix, rang in (0, 128)
    :type prefix: int
    :return: comma separated ipv6 net mask code,
             eg: ffff:ffff:ffff:ffff:0000:0000:0000:0000
    :rtype: str
    """
    if prefix > 128 or prefix < 0:
        raise ValueError("invalid cidr prefix for ipv6")
    else:
        mask = ((1 << 128) - 1) ^ ((1 << (128 - prefix)) - 1)
        f = 15  # 0xf or 0b1111
        hex_mask_str = ''
        for i in range(0, 32):
            hex_mask_str = format((mask & f), 'x') + hex_mask_str
            mask = mask >> 4
            if i != 31 and i & 3 == 3:
                hex_mask_str = ':' + hex_mask_str
        return hex_mask_str
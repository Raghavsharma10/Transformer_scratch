def ipv4_prefix_to_mask(prefix):
    """
    ipv4 cidr prefix to net mask

    :param prefix: cidr prefix , rang in (0, 32)
    :type prefix: int
    :return: dot separated ipv4 net mask code, eg: 255.255.255.0
    :rtype: str
    """
    if prefix > 32 or prefix < 0:
        raise ValueError("invalid cidr prefix for ipv4")
    else:
        mask = ((1 << 32) - 1) ^ ((1 << (32 - prefix)) - 1)
        eight_ones = 255  # 0b11111111
        mask_str = ''
        for i in range(0, 4):
            mask_str = str(mask & eight_ones) + mask_str
            mask = mask >> 8
            if i != 3:
                mask_str = '.' + mask_str
        return mask_str
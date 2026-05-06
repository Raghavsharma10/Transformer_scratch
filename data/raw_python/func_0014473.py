def _generate_prime(bits, rng):
    "primtive attempt at prime generation"
    hbyte_mask = pow(2, bits % 8) - 1
    while True:
        # loop catches the case where we increment n into a higher bit-range
        x = rng.read((bits+7) // 8)
        if hbyte_mask > 0:
            x = chr(ord(x[0]) & hbyte_mask) + x[1:]
        n = util.inflate_long(x, 1)
        n |= 1
        n |= (1 << (bits - 1))
        while not number.isPrime(n):
            n += 2
        if util.bit_length(n) == bits:
            break
    return n
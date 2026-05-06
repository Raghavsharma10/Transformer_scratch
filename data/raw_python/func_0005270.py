def iso_payment_reference_validator(v: str):
    """
    Validates ISO reference number checksum.
    :param v: Reference number
    """
    num = ''
    v = STRIP_WHITESPACE.sub('', v)
    for ch in v[4:] + v[0:4]:
        x = ord(ch)
        if ord('0') <= x <= ord('9'):
            num += ch
        else:
            x -= 55
            if x < 10 or x > 35:
                raise ValidationError(_('Invalid payment reference: {}').format(v))
            num += str(x)
    res = Decimal(num) % Decimal('97')
    if res != Decimal('1'):
        raise ValidationError(_('Invalid payment reference: {}').format(v))
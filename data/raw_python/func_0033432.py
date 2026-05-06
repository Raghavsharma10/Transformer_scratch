def calculate_checksum(isbn):
    """Calculate ISBN checksum.

    Args:
        isbn (str): SBN, ISBN-10 or ISBN-13

    Returns:
        ``str``: Checksum for given ISBN or SBN

    """
    isbn = [int(i) for i in _isbn_cleanse(isbn, checksum=False)]
    if len(isbn) == 9:
        products = [x * y for x, y in zip(isbn, range(1, 10))]
        check = sum(products) % 11
        if check == 10:
            check = 'X'
    else:
        # As soon as Python 2.4 support is dumped
        # [(isbn[i] if i % 2 == 0 else isbn[i] * 3) for i in range(12)]
        products = []
        for i in range(12):
            if i % 2 == 0:
                products.append(isbn[i])
            else:
                products.append(isbn[i] * 3)
        check = 10 - sum(products) % 10
        if check == 10:
            check = 0
    return str(check)
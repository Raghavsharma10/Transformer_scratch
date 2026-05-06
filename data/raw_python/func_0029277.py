def is_credit_card(string, card_type=None):
    """
    Checks if a string is a valid credit card number.
    If card type is provided then it checks that specific type,
    otherwise any known credit card number will be accepted.

    :param string: String to check.
    :type string: str
    :param card_type: Card type.
    :type card_type: str

    Can be one of these:

    * VISA
    * MASTERCARD
    * AMERICAN_EXPRESS
    * DINERS_CLUB
    * DISCOVER
    * JCB

    or None. Default to None (any card).

    :return: True if credit card, false otherwise.
    :rtype: bool
    """
    if not is_full_string(string):
        return False
    if card_type:
        if card_type not in CREDIT_CARDS:
            raise KeyError(
                'Invalid card type "{}". Valid types are: {}'.format(card_type, ', '.join(CREDIT_CARDS.keys()))
            )
        return bool(CREDIT_CARDS[card_type].search(string))
    for c in CREDIT_CARDS:
        if CREDIT_CARDS[c].search(string):
            return True
    return False
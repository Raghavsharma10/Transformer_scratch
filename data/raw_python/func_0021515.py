def convert_money(amount, currency_from, currency_to):
    """
    Convert 'amount' from 'currency_from' to 'currency_to' and return a Money
    instance of the converted amount.
    """
    new_amount = base_convert_money(amount, currency_from, currency_to)
    return moneyed.Money(new_amount, currency_to)
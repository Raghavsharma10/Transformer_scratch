def numprop(value, propertyexpected):
    """
    Check If A Number Is A Type
    """
    if propertyexpected == 'triangular':
        x = (math.sqrt(8 * value + 1) - 1) / 2
        return bool(x - int(x) > 0)
    elif propertyexpected == 'square':
        return math.sqrt(value).is_integer()
    elif propertyexpected == 'cube':
        x = value**(1 / 3)
        x = int(round(x))
        return bool(x**3 == value)
    elif propertyexpected == 'even':
        return value % 2 == 0
    elif propertyexpected == 'odd':
        return not value % 2 == 0
    elif propertyexpected == 'positive':
        return bool(value > 0)
    elif propertyexpected == 'negative':
        return bool(value < 0)
    elif propertyexpected == 'zero':
        return bool(value == 0)
def between(min_value, max_value):
    'Numerical values limit'
    message = N_('value should be between %(min)d and %(max)d') % \
                    dict(min=min_value, max=max_value)

    @validator(message)
    def wrapper(conv, value):
        if value is None:
            # it meens that this value is not required
            return True
        if value < min_value:
            return False
        if value > max_value:
            return False
        return True
    return wrapper
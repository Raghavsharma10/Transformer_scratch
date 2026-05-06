def gt(gt_value):
    """
    Validates that a field value is greater than the
    value given to this validator.
    """
    def validate(value):
        if value <= gt_value:
            return e("{} is not greater than {}", value, gt_value)
    return validate
def each_item(*validators):
    """
    A wrapper which applies the given validators to each item in a field
    value of type `list`.

    Example usage in a Schema:

    "my_list_field": {"type": Array(int), "validates": each_item(lte(10))}
    """
    def validate(value):
        for item in value:
            for validator in validators:
                error = validator(item)
                if error:
                    return error
        return None
    return validate
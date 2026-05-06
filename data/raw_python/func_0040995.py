def describe(cls, full=False):
    """Prints a description of the table based on the provided
    documentation and post processors"""
    divider_double = "=" * 80
    divider_single = "-" * 80
    description = cls.__doc__
    message = []
    message.append(divider_double)
    message.append(cls.__name__ + ':')
    message.append(description)
    if full and cls.post_processors(cls):
        message.append(divider_single)
        message.append("Post processors:")
        message.append(divider_single)
        for processor in cls.post_processors(cls):
            message.append(">" + " " * 3 + processor.__name__ + ':')
            message.append(" " * 4 + processor.__doc__)
            message.append('')
    message.append(divider_double)
    message.append('')
    for line in message:
        print(line)
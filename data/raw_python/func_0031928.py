def _sanitize_numbers(uncleaned_numbers):
    """
        Convert strings to integers if possible
    """
    cleaned_numbers = []
    for x in uncleaned_numbers:
        try:
            cleaned_numbers.append(int(x))
        except ValueError:
            cleaned_numbers.append(x)
    return cleaned_numbers
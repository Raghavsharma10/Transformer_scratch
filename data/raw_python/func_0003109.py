def token_at_cursor(code, pos=0):
    """
    Find the token present at the passed position in the code buffer
     :return (tuple): a pair (token, start_position)
    """
    l = len(code)
    end = start = pos
    # Go forwards while we get alphanumeric chars
    while end < l and code[end].isalpha():
        end += 1
    # Go backwards while we get alphanumeric chars
    while start > 0 and code[start-1].isalpha():
        start -= 1
    # If previous character is a %, add it (potential magic)
    if start > 0 and code[start-1] == '%':
        start -= 1
    return code[start:end], start
def make_unique_str(num_chars=20):
    """make a random string of characters for a temp filename"""
    chars = 'abcdefghigklmnopqrstuvwxyz'
    all_chars = chars + chars.upper() + '01234567890'
    picks = list(all_chars)
    return ''.join([choice(picks) for i in range(num_chars)])
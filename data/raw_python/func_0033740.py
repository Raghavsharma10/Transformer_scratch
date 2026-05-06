def format_header_cell(val):
    """
    Formats given header column. This involves changing '_Px_' to '(', '_xP_' to ')' and
    all other '_' to spaces.
    """
    return re.sub('_', ' ', re.sub(r'(_Px_)', '(', re.sub(r'(_xP_)', ')', str(val) )))
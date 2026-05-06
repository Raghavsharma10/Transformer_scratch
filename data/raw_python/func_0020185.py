def delimiters_to_re(delimiters):
    """convert delimiters to corresponding regular expressions"""

    # caching
    delimiters = tuple(delimiters)
    if delimiters in re_delimiters:
        re_tag = re_delimiters[delimiters]
    else:
        open_tag, close_tag = delimiters

        # escape
        open_tag = ''.join([c if c.isalnum() else '\\' + c for c in open_tag])
        close_tag = ''.join([c if c.isalnum() else '\\' + c for c in close_tag])

        re_tag = re.compile(open_tag + r'([#^>&{/!=]?)\s*(.*?)\s*([}=]?)' + close_tag, re.DOTALL)
        re_delimiters[delimiters] = re_tag

    return re_tag
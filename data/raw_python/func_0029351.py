def get_response(_code):
    """
    Return xx1x response for xx0x codes (e.g. 0810 for 0800)
    """
    if _code:
        code = str(_code)
        return code[:-2] + str(int(code[-2:-1]) + 1) + code[-1]
    else:
        return None
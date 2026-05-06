def p_retry(p):
    """
    retry : RETRY OPEN_CURLY_BRACKET action_list CLOSE_CURLY_BRACKET
          | RETRY OPEN_BRACKET retry_option_list CLOSE_BRACKET OPEN_CURLY_BRACKET action_list CLOSE_CURLY_BRACKET
    """
    if len(p) == 5:
        p[0] = Retry(p[3])
    elif len(p) == 8:
        p[0] = Retry(p[6], **p[3])
    else:
        raise RuntimeError("Invalid product rules for 'retry_option_list'")
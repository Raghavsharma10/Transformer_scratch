def _search_regex(ops: dict, regex_pat: str):
    """
    Search order:
      * specified regexps
      * operators sorted from longer to shorter
    """
    custom_regexps = list(filter(None, [dic['regex'] for op, dic in ops.items() if 'regex' in dic]))
    op_names = [op for op, dic in ops.items() if 'regex' not in dic]
    regex = [regex_pat.format(_ops_regex(op_names))] if len(op_names) > 0 else []
    return re.compile('|'.join(custom_regexps + regex))
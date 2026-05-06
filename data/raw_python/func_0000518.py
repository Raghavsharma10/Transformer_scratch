def transform_source(text):
    '''removes a "where" clause which is identified by the use of "where"
    as an identifier and ends at the first DEDENT (i.e. decrease in indentation)'''
    toks = tokenize.generate_tokens(StringIO(text).readline)
    result = []
    where_clause = False
    for toktype, tokvalue, _, _, _ in toks:
        if toktype == tokenize.NAME and tokvalue == "where":
            where_clause = True
        elif where_clause and toktype == tokenize.DEDENT:
            where_clause = False
            continue

        if not where_clause:
            result.append((toktype, tokvalue))
    return tokenize.untokenize(result)
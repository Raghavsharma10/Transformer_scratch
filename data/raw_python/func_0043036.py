def _esfilter2sqlwhere(db, esfilter):
    """
    CONVERT ElassticSearch FILTER TO SQL FILTER
    db - REQUIRED TO PROPERLY QUOTE VALUES AND COLUMN NAMES
    """
    esfilter = wrap(esfilter)

    if esfilter is True:
        return SQL_TRUE
    elif esfilter["and"]:
        return sql_iso(SQL_AND.join([esfilter2sqlwhere(db, a) for a in esfilter["and"]]))
    elif esfilter["or"]:
        return sql_iso(SQL_OR.join([esfilter2sqlwhere(db, a) for a in esfilter["or"]]))
    elif esfilter["not"]:
        return SQL_NOT + sql_iso(esfilter2sqlwhere(db, esfilter["not"]))
    elif esfilter.term:
        return sql_iso(SQL_AND.join([
            quote_column(col) + SQL("=") + quote_value(val)
            for col, val in esfilter.term.items()
        ]))
    elif esfilter.terms:
        for col, v in esfilter.terms.items():
            if len(v) == 0:
                return "FALSE"

            try:
                int_list = convert.value2intlist(v)
                has_null = False
                for vv in v:
                    if vv == None:
                        has_null = True
                        break
                if int_list:
                    filter = int_list_packer(col, int_list)
                    if has_null:
                        return esfilter2sqlwhere(db, {"or": [{"missing": col}, filter]})
                    elif 'terms' in filter and set(filter['terms'].get(col, []))==set(int_list):
                        return quote_column(col) + " in " + quote_list(int_list)
                    else:
                        return esfilter2sqlwhere(db, filter)
                else:
                    if has_null:
                        return esfilter2sqlwhere(db, {"missing": col})
                    else:
                        return "false"
            except Exception as e:
                e = Except.wrap(e)
                pass
            return quote_column(col) + " in " + quote_list(v)
    elif esfilter.script:
        return sql_iso(esfilter.script)
    elif esfilter.range:
        name2sign = {
            "gt": SQL(">"),
            "gte": SQL(">="),
            "lte": SQL("<="),
            "lt": SQL("<")
        }

        def single(col, r):
            min = coalesce(r["gte"], r[">="])
            max = coalesce(r["lte"], r["<="])
            if min != None and max != None:
                # SPECIAL CASE (BETWEEN)
                sql = quote_column(col) + SQL(" BETWEEN ") + quote_value(min) + SQL_AND + quote_value(max)
            else:
                sql = SQL_AND.join(
                    quote_column(col) + name2sign[sign] + quote_value(value)
                    for sign, value in r.items()
                )
            return sql

        terms = [single(col, ranges) for col, ranges in esfilter.range.items()]
        if len(terms) == 1:
            output = terms[0]
        else:
            output = sql_iso(SQL_AND.join(terms))
        return output
    elif esfilter.missing:
        if isinstance(esfilter.missing, text_type):
            return sql_iso(quote_column(esfilter.missing) + SQL_IS_NULL)
        else:
            return sql_iso(quote_column(esfilter.missing.field) + SQL_IS_NULL)
    elif esfilter.exists:
        if isinstance(esfilter.exists, text_type):
            return sql_iso(quote_column(esfilter.exists) + SQL_IS_NOT_NULL)
        else:
            return sql_iso(quote_column(esfilter.exists.field) + SQL_IS_NOT_NULL)
    elif esfilter.match_all:
        return SQL_TRUE
    elif esfilter.instr:
        return sql_iso(SQL_AND.join(["instr" + sql_iso(quote_column(col) + ", " + quote_value(val)) + ">0" for col, val in esfilter.instr.items()]))
    else:
        Log.error("Can not convert esfilter to SQL: {{esfilter}}", esfilter=esfilter)
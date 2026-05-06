def find_holes(db_module, db, table_name, column_name, _range, filter=None):
    """
    FIND HOLES IN A DENSE COLUMN OF INTEGERS
    RETURNS A LIST OF {"min"min, "max":max} OBJECTS
    """
    if not filter:
        filter = {"match_all": {}}

    _range = wrap(_range)
    params = {
        "min": _range.min,
        "max": _range.max - 1,
        "column_name": db_module.quote_column(column_name),
        "table_name": db_module.quote_column(table_name),
        "filter": esfilter2sqlwhere(filter)
    }

    min_max = db.query("""
        SELECT
            min({{column_name}}) `min`,
            max({{column_name}})+1 `max`
        FROM
            {{table_name}} a
        WHERE
            a.{{column_name}} BETWEEN {{min}} AND {{max}} AND
            {{filter}}
    """, params)[0]

    db.execute("SET @last={{min}}-1", {"min": _range.min})
    ranges = db.query("""
        SELECT
            prev_rev+1 `min`,
            curr_rev `max`
        FROM (
            SELECT
                a.{{column_name}}-@last diff,
                @last prev_rev,
                @last:=a.{{column_name}} curr_rev
            FROM
                {{table_name}} a
            WHERE
                a.{{column_name}} BETWEEN {{min}} AND {{max}} AND
                {{filter}}
            ORDER BY
                a.{{column_name}}
        ) a
        WHERE
            diff>1
    """, params)

    if ranges:
        ranges.append({"min": min_max.max, "max": _range.max})
    else:
        if min_max.min:
            ranges.append({"min": _range.min, "max": min_max.min})
            ranges.append({"min": min_max.max, "max": _range.max})
        else:
            ranges.append(_range)

    return ranges
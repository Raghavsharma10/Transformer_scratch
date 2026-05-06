def _where_terms(master, where, schema):
    """
    USE THE SCHEMA TO CONVERT DIMENSION NAMES TO ES FILTERS
    master - TOP LEVEL WHERE (FOR PLACING NESTED FILTERS)
    """
    if is_data(where):
        if where.term:
            # MAP TERM
            try:
                output = _map_term_using_schema(master, [], where.term, schema.edges)
                return output
            except Exception as e:
                Log.error("programmer problem?", e)
        elif where.terms:
            # MAP TERM
            output = FlatList()
            for k, v in where.terms.items():
                if not is_container(v):
                    Log.error("terms filter expects list of values")
                edge = schema.edges[k]
                if not edge:
                    output.append({"terms": {k: v}})
                else:
                    if is_text(edge):
                        # DIRECT FIELD REFERENCE
                        return {"terms": {edge: v}}
                    try:
                        domain = edge.getDomain()
                    except Exception as e:
                        Log.error("programmer error", e)
                    fields = domain.dimension.fields
                    if is_data(fields):
                        or_agg = []
                        for vv in v:
                            and_agg = []
                            for local_field, es_field in fields.items():
                                vvv = vv[local_field]
                                if vvv != None:
                                    and_agg.append({"term": {es_field: vvv}})
                            or_agg.append({"and": and_agg})
                        output.append({"or": or_agg})
                    elif is_list(fields) and len(fields) == 1 and is_variable_name(fields[0]):
                        output.append({"terms": {fields[0]: v}})
                    elif domain.partitions:
                        output.append({"or": [domain.getPartByKey(vv).esfilter for vv in v]})
            return {"and": output}
        elif where["or"]:
            return {"or": [unwrap(_where_terms(master, vv, schema)) for vv in where["or"]]}
        elif where["and"]:
            return {"and": [unwrap(_where_terms(master, vv, schema)) for vv in where["and"]]}
        elif where["not"]:
            return {"not": unwrap(_where_terms(master, where["not"], schema))}
    return where
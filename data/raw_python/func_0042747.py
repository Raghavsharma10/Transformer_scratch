def accumulate_nested_doc(nested_path, expr=IDENTITY):
    """
    :param nested_path: THE PATH USED TO EXTRACT THE NESTED RECORDS
    :param expr: FUNCTION USED ON THE NESTED OBJECT TO GET SPECIFIC VALUE
    :return: THE DE_TYPED NESTED OBJECT ARRAY
    """
    name = literal_field(nested_path)
    def output(doc):
        acc = []
        for h in doc.inner_hits[name].hits.hits:
            i = h._nested.offset
            obj = Data()
            for f, v in h.fields.items():
                local_path = untype_path(relative_field(f, nested_path))
                obj[local_path] = unwraplist(v)
            # EXTEND THE LIST TO THE LENGTH WE REQUIRE
            for _ in range(len(acc), i+1):
                acc.append(None)
            acc[i] = expr(obj)
        return acc
    return output
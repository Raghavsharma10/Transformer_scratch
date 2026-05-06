def diff_schema(A, B):
    """
    RETURN PROPERTIES IN A, BUT NOT IN B
    :param A: elasticsearch properties
    :param B: elasticsearch properties
    :return: (name, properties) PAIRS WHERE name IS DOT-DELIMITED PATH
    """
    output =[]
    def _diff_schema(path, A, B):
        for k, av in A.items():
            if k == "_id" and path == ".":
                continue  # DO NOT ADD _id TO ANY SCHEMA DIFF
            bv = B[k]
            if bv == None:
                output.append((concat_field(path, k), av))
            elif av.type == bv.type:
                pass  # OK
            elif (av.type == None and bv.type == 'object') or (av.type == 'object' and bv.type == None):
                pass  # OK
            else:
                Log.warning("inconsistent types: {{typeA}} vs {{typeB}}", typeA=av.type, typeB=bv.type)
            _diff_schema(concat_field(path, k), av.properties, bv.properties)

    # what to do with conflicts?
    _diff_schema(".", A, B)
    return output
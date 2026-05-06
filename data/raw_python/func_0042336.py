def map2set(data, relation):
    """
    EXPECTING A is_data(relation) THAT MAPS VALUES TO lists
    THE LISTS ARE EXPECTED TO POINT TO MEMBERS OF A SET
    A set() IS RETURNED
    """
    if data == None:
        return Null
    if isinstance(relation, Data):
        Log.error("Does not accept a Data")

    if is_data(relation):
        try:
            # relation[d] is expected to be a list
            # return set(cod for d in data for cod in relation[d])
            output = set()
            for d in data:
                for cod in relation.get(d, []):
                    output.add(cod)
            return output
        except Exception as e:
            Log.error("Expecting a dict with lists in codomain", e)
    else:
        try:
            # relation[d] is expected to be a list
            # return set(cod for d in data for cod in relation[d])
            output = set()
            for d in data:
                cod = relation(d)
                if cod == None:
                    continue
                output.add(cod)
            return output
        except Exception as e:
            Log.error("Expecting a dict with lists in codomain", e)
    return Null
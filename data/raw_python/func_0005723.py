def show_faults():
    """
    Return all valid/active faults ordered by ID to allow the user to pick and choose.

    :return:  List of Tuples where the Tuple elements are:  (fault id, fault template)
    """
    cursor = CONN.cursor()

    query = "select fau_id, fault from surfaults where fau_is_valid = 'y' order by fau_id asc"
    cursor.execute(query)
    result = cursor.fetchall()
    return result
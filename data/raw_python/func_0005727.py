def __get_sentence(counts, sentence_id=None):
    """Let's fetch a random sentence that we then need to substitute bits of...
    @
    :param counts:
    :param sentence_id:
    """

    # First of all we need a cursor and a query to retrieve our ID's
    cursor = CONN.cursor()
    check_query = "select sen_id from sursentences"

    # Now we fetch the result of the query and save it into check_result
    cursor.execute(check_query)
    check_result = cursor.fetchall()

    # declare an empty list to be populated below
    id_list = []
    id_to_fetch = None

    # Populate the id_list variable with all of the ID's we retrieved from the database query.
    for row in check_result:
        id_list.append(row[0])

    if sentence_id is not None:
        if type(sentence_id) is int:
            id_to_fetch = sentence_id
    else:
        id_to_fetch = random.randint(1, counts['max_sen'])

        while id_to_fetch not in id_list:
            id_to_fetch = random.randint(1, counts['max_sen'])

    query = ("select * from sursentences where sen_id = {0}".format(id_to_fetch))
    cursor.execute(query)
    result = cursor.fetchone()
    # cursor.close()

    return result
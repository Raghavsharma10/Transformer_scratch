def show_sentences():
    """
    Return all valid/active sentences ordered by ID to allow the user to pick and choose.

    :return:  Dict containing the sentence ID as the key and the sentence structure as the value.
    """
    cursor = CONN.cursor()

    query = "select sen_id, sentence from sursentences where sen_is_valid = 'y' order by sen_id asc"
    cursor.execute(query)
    result = cursor.fetchall()

    response_dict = {}

    for row in result:
        response_dict[row[0]] = row[1]

    return response_dict
def offer_answer(pool, answer, rationale, student_id, algo, options):
    """
    submit a student answer to the answer pool

    The answer maybe selected to stay in the pool depending on the selection algorithm

    Args:
        pool (dict): answer pool
            Answer pool format:
            {
                option1_index: {
                    'student_id': { can store algorithm specific info here },
                    ...
                }
                option2_index: ...
            }
        answer (int): the option student selected
        rationale (str): the rationale text
        student_id (str): student identifier
        algo (str): the selection algorithm
        options (dict): the options available in the question

    Raises:
        UnknownChooseAnswerAlgorithm: when we don't know the algorithm
    """
    if algo['name'] == 'simple':
        offer_simple(pool, answer, rationale, student_id, options)
    elif algo['name'] == 'random':
        offer_random(pool, answer, rationale, student_id, options)
    else:
        raise UnknownChooseAnswerAlgorithm()
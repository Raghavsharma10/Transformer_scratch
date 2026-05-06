def get_other_answers(pool, seeded_answers, get_student_item_dict, algo, options):
    """
    Select other student's answers from answer pool or seeded answers based on the selection algorithm

    Args:
        pool (dict): answer pool, format:
            {
                option1_index: {
                    student_id: { can store algorithm specific info here }
                },
                option2_index: {
                    student_id: { ... }
                }
            }
        seeded_answers (list): seeded answers from instructor
            [
                {'answer': 0, 'rationale': 'rationale A'},
                {'answer': 1, 'rationale': 'rationale B'},
            ]
        get_student_item_dict (callable): get student item dict function to return student item dict
        algo (str): selection algorithm
        options (dict): answer options for the question

    Returns:
        dict: answers based on the selection algorithm
    """
    # "#" means the number of responses returned should be the same as the number of options.
    num_responses = len(options) \
        if 'num_responses' not in algo or algo['num_responses'] == "#" \
        else int(algo['num_responses'])

    if algo['name'] == 'simple':
        return get_other_answers_simple(pool, seeded_answers, get_student_item_dict, num_responses)
    elif algo['name'] == 'random':
        return get_other_answers_random(pool, seeded_answers, get_student_item_dict, num_responses)
    else:
        raise UnknownChooseAnswerAlgorithm()
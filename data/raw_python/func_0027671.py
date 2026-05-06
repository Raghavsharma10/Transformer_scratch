def validate_seeded_answers(answers, options, algo):
    """
    Validate answers based on selection algorithm

    This is called when instructor setup the tool and providing seeded answers to the question.
    This function is trying to validate if instructor provided enough seeds for a give algorithm.
    e.g. we require 1 seed for each option in simple algorithm and at least 1 seed for random
    algorithm. Because otherwise, the first student won't be able to see the answers on the
    second step where he/she suppose to compare and review other students answers.

    Args:
        answers (list): list of dict that contain seeded answers
        options (dict): all options that should exist in the answers
        algo (str): selection algorithm

    Returns:
        None if successful, otherwise error message
    """
    if algo['name'] == 'simple':
        return validate_seeded_answers_simple(answers, options, algo)
    elif algo['name'] == 'random':
        return validate_seeded_answers_random(answers)
    else:
        raise UnknownChooseAnswerAlgorithm()
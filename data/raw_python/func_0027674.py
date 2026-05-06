def get_other_answers_random(pool, seeded_answers, get_student_item_dict, num_responses):
    """
    Get answers from others with random algorithm, which randomly select answer from the pool.

    Student may get three answers for option 1 or one answer for option 1 and two answers for option 2.

    Args:
        see `get_other_answers`
        num_responses (int): the number of responses to be returned. This value may not be
            respected if there is not enough answers to return

    Returns:
        dict: answers based on the selection algorithm
    """
    ret = []
    # clean up answers so that all keys are int
    pool = {int(k): v for k, v in pool.items()}
    seeded = {'seeded'+str(index): answer for index, answer in enumerate(seeded_answers)}
    merged_pool = seeded.keys()

    for key in pool:
        merged_pool += pool[key].keys()

    # shuffle
    random.shuffle(merged_pool)
    # get student identifier
    student_id = get_student_item_dict()['student_id']

    for student in merged_pool:
        if len(ret) >= num_responses:
            # have enough answers
            break
        elif student == student_id:
            # this is the student's answer so don't return
            continue

        if student.startswith('seeded'):
            option = seeded[student]['answer']
            rationale = seeded[student]['rationale']
        else:
            student_item = get_student_item_dict(student)
            submission = sas_api.get_answers_for_student(student_item)
            rationale = submission.get_rationale(0)
            option = submission.get_vote(0)
        ret.append({'option': option, 'rationale': rationale})

    return {"answers": ret}
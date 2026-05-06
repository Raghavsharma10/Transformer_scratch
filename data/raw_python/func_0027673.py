def get_other_answers_simple(pool, seeded_answers, get_student_item_dict, num_responses):
    """
    Get answers from others with simple algorithm, which picks one answer for each option.

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
    total_in_pool = len(seeded_answers)
    merged_pool = convert_seeded_answers(seeded_answers)
    student_id = get_student_item_dict()['student_id']
    # merge the dictionaries in the answer dictionary
    for key in pool:
        total_in_pool += len(pool[key])
        # if student_id has value, we assume the student just submitted an answer. So removing it
        # from total number in the pool
        if student_id in pool[key].keys():
            total_in_pool -= 1
        if key in merged_pool:
            merged_pool[key].update(pool[key].items())
        else:
            merged_pool[key] = pool[key]

    # remember which option+student_id is selected, so that we don't have duplicates in the result
    selected = []

    # loop until we have enough answers to return
    while len(ret) < min(num_responses, total_in_pool):
        for option, students in merged_pool.items():
            student = student_id
            i = 0
            while (student == student_id or i > 100) and (str(option) + student) not in selected:
                # retry until we got a different one or after 100 retries
                # we are suppose to get a different student answer or a seeded one in a few tries
                # as we have at least one seeded answer for each option in the algo. And it is not
                # suppose to overflow i order to break the loop
                student = random.choice(students.keys())
                i += 1
            selected.append(str(option)+student)
            if student.startswith('seeded'):
                # seeded answer, get the rationale from local
                rationale = students[student]
            else:
                student_item = get_student_item_dict(student)
                submission = sas_api.get_answers_for_student(student_item)
                rationale = submission.get_rationale(0)
            ret.append({'option': option, 'rationale': rationale})

            # check if we have enough answers
            if len(ret) >= min(num_responses, total_in_pool):
                break

    return {"answers": ret}
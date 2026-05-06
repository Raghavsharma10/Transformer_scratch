def add_answer_for_student(student_item, vote, rationale):
    """
    Add an answer for a student to the backend

    Args:
        student_item (dict): The location of the problem this submission is
            associated with, as defined by a course, student, and item.
        vote (int): the option that student voted for
        rationale (str): the reason why the student vote for the option
    """
    answers = get_answers_for_student(student_item)
    answers.add_answer(vote, rationale)

    sub_api.create_submission(student_item, {
        ANSWER_LIST_KEY: answers.get_answers_as_list()
    })
def get_answers_for_student(student_item):
    """
    Retrieve answers from backend for a student and question

    Args:
        student_item (dict): The location of the problem this submission is
            associated with, as defined by a course, student, and item.

    Returns:
        Answers: answers for the student
    """
    submissions = sub_api.get_submissions(student_item)
    if not submissions:
        return Answers()

    latest_submission = submissions[0]
    latest_answer_item = latest_submission.get('answer', {})
    return Answers(latest_answer_item.get(ANSWER_LIST_KEY, []))
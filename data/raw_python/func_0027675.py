def convert_seeded_answers(answers):
    """
    Convert seeded answers into the format that can be merged into student answers.

    Args:
        answers (list): seeded answers

    Returns:
        dict: seeded answers with student answers format:
            {
                0: {
                    'seeded0': 'rationaleA'
                }
                1: {
                    'seeded1': 'rationaleB'
                }
            }
    """
    converted = {}
    for index, answer in enumerate(answers):
        converted.setdefault(answer['answer'], {})
        converted[answer['answer']]['seeded' + str(index)] = answer['rationale']

    return converted
def score_x_of_a_kind_yatzy(dice: List[int], min_same_faces: int) -> int:
    """Similar to yahtzee, but only return the sum of the dice that satisfy min_same_faces
    """
    for die, count in Counter(dice).most_common(1):
        if count >= min_same_faces:
            return die * min_same_faces
    return 0
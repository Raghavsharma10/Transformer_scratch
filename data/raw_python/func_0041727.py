def score_x_of_a_kind_yahtzee(dice: List[int], min_same_faces: int) -> int:
    """Return sum of dice if there are a minimum of equal min_same_faces dice, otherwise
    return zero. Only works for 3 or more min_same_faces.
    """
    for die, count in Counter(dice).most_common(1):
        if count >= min_same_faces:
            return sum(dice)
    return 0
def score_small_straight_yatzy(dice: List[int]) -> int:
    """
    Small straight scoring according to yatzy rules
    """
    dice_set = set(dice)
    if _are_two_sets_equal({1, 2, 3, 4, 5}, dice_set):
        return sum(dice)
    return 0
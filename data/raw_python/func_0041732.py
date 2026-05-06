def score_large_straight_yatzy(dice: List[int]) -> int:
    """
    Large straight scoring according to yatzy rules
    """
    dice_set = set(dice)
    if _are_two_sets_equal({2, 3, 4, 5, 6}, dice_set):
        return sum(dice)
    return 0
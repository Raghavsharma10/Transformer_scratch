def score_large_straight_yahtzee(dice: List[int]) -> int:
    """
    Large straight scoring according to regular yahtzee rules
    """
    global CONSTANT_SCORES_YAHTZEE
    dice_set = set(dice)
    if _are_two_sets_equal({1, 2, 3, 4, 5}, dice_set) or \
            _are_two_sets_equal({2, 3, 4, 5, 6}, dice_set):
        return CONSTANT_SCORES_YAHTZEE[Category.LARGE_STRAIGHT]
    return 0
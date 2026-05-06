def score_small_straight_yahztee(dice: List[int]) -> int:
    """
    Small straight scoring according to regular yahtzee rules
    """
    global CONSTANT_SCORES_YAHTZEE
    dice_set = set(dice)
    if _are_two_sets_equal({1, 2, 3, 4}, dice_set) or \
            _are_two_sets_equal({2, 3, 4, 5}, dice_set) or \
            _are_two_sets_equal({3, 4, 5, 6}, dice_set):
        return CONSTANT_SCORES_YAHTZEE[Category.SMALL_STRAIGHT]
    return 0
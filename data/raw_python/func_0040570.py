def _is_valid_options_weights_list(value):
    '''Check whether ``values`` is a valid argument for ``weighted_choice``.'''
    return ((isinstance(value, list)) and
            len(value) > 1 and
            (all(isinstance(opt, tuple) and
                 len(opt) == 2 and
                 isinstance(opt[1], (int, float))
                 for opt in value)))
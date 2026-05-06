def step_random_processes(oscillators):
    """
    Args:
        oscillators (list): A list of oscillator.Oscillator objects
            to operate on

    Returns: None
    """
    if not rand.prob_bool(0.01):
        return
    amp_bias_weights = [(0.001, 1), (0.1, 100), (0.15, 40), (1, 0)]
    # Find out how many oscillators should move
    num_moves = iching.get_hexagram('NAIVE') % len(oscillators)
    for i in range(num_moves):
        pair = [gram % len(oscillators)
                for gram in iching.get_hexagram('THREE COIN')]
        amplitudes = [(gram / 64) * rand.weighted_rand(amp_bias_weights)
                      for gram in iching.get_hexagram('THREE COIN')]
        oscillators[pair[0]].amplitude.drift_target = amplitudes[0]
        oscillators[pair[1]].amplitude.drift_target = amplitudes[1]
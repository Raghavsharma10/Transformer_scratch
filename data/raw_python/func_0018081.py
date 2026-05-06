def linear_scheduler_up(init_value, target_value, duration):
    """ Increases linearly and then stays flat """

    value = init_value
    t = 0
    while True:
        yield value
        t += 1
        if t < duration:
            value = init_value + t * (target_value - init_value) / duration
        else:
            value = target_value
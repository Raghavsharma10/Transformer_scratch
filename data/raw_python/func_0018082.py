def linear_scheduler_up_down(init_value, target_value, final_value,
                             duration_up, t_decrease, duration_down):
    """ Increases linearly to target_value, stays at target_value until
    t_decrease and then decreases linearly
    """

    value = init_value
    t = 0

    while True:
        yield value
        t += 1
        if t < duration_up:
            value = init_value + t * (target_value - init_value) / \
                    float(duration_up)
        elif t > t_decrease:
            value = target_value - (t - t_decrease) * \
                    (target_value - final_value) / \
                    float(duration_down)
        else:
            value = target_value
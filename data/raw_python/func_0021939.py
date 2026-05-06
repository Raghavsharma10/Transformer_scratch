def hide_ticks(plot, min_tick_value=None, max_tick_value=None):
    """Hide tick values that are outside of [min_tick_value, max_tick_value]"""
    for tick, tick_value in zip(plot.get_yticklabels(), plot.get_yticks()):
        tick_label = as_numeric(tick_value)
        if tick_label:
            if (min_tick_value is not None and tick_label < min_tick_value or 
                 max_tick_value is not None and tick_label > max_tick_value):
                tick.set_visible(False)
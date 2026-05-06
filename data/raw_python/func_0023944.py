def arrow_respond(slider, event):
    """Event handler for arrow key events in plot windows.
    
    Pass the slider object to update as a masked argument using a lambda function::
        
        lambda evt: arrow_respond(my_slider, evt)
    
    Parameters
    ----------
    slider : Slider instance associated with this handler.
    event : Event to be handled.
    """
    if event.key == 'right':
        slider.set_val(min(slider.val + 1, slider.valmax))
    elif event.key == 'left':
        slider.set_val(max(slider.val - 1, slider.valmin))
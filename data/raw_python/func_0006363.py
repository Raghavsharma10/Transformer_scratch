def get_events_with_error_code(event_number, event_status, select_mask=0b1111111111111111, condition=0b0000000000000000):
    '''Selects the events with a certain error code.

    Parameters
    ----------
    event_number : numpy.array
    event_status : numpy.array
    select_mask : int
        The mask that selects the event error code to check.
    condition : int
        The value the selected event error code should have.

    Returns
    -------
    numpy.array
    '''

    logging.debug("Calculate events with certain error code")
    return np.unique(event_number[event_status & select_mask == condition])
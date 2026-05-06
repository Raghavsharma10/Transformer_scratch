def check_bad_data(raw_data, prepend_data_headers=None, trig_count=None):
    """Checking FEI4 raw data array for corrupted data.
    """
    consecutive_triggers = 16 if trig_count == 0 else trig_count
    is_fe_data_header = logical_and(is_fe_word, is_data_header)
    trigger_idx = np.where(is_trigger_word(raw_data) >= 1)[0]
    fe_dh_idx = np.where(is_fe_data_header(raw_data) >= 1)[0]
    n_triggers = trigger_idx.shape[0]
    n_dh = fe_dh_idx.shape[0]

    # get index of the last trigger
    if n_triggers:
        last_event_data_headers_cnt = np.where(fe_dh_idx > trigger_idx[-1])[0].shape[0]
        if consecutive_triggers and last_event_data_headers_cnt == consecutive_triggers:
            if not np.all(trigger_idx[-1] > fe_dh_idx):
                trigger_idx = np.r_[trigger_idx, raw_data.shape]
            last_event_data_headers_cnt = None
        elif last_event_data_headers_cnt != 0:
            fe_dh_idx = fe_dh_idx[:-last_event_data_headers_cnt]
        elif not np.all(trigger_idx[-1] > fe_dh_idx):
            trigger_idx = np.r_[trigger_idx, raw_data.shape]
    # if any data header, add trigger for histogramming, next readout has to have trigger word
    elif n_dh:
        trigger_idx = np.r_[trigger_idx, raw_data.shape]
        last_event_data_headers_cnt = None
    # no trigger, no data header
    # assuming correct data, return input values
    else:
        return False, prepend_data_headers, n_triggers, n_dh

#     # no triggers, check for the right amount of data headers
#     if consecutive_triggers and prepend_data_headers and prepend_data_headers + n_dh != consecutive_triggers:
#         return True, n_dh, n_triggers, n_dh

    n_triggers_cleaned = trigger_idx.shape[0]
    n_dh_cleaned = fe_dh_idx.shape[0]

    # check that trigger comes before data header
    if prepend_data_headers is None and n_triggers_cleaned and n_dh_cleaned and not trigger_idx[0] < fe_dh_idx[0]:
        return True, last_event_data_headers_cnt, n_triggers, n_dh  # FIXME: 0?
    # check that no trigger comes before the first data header
    elif consecutive_triggers and prepend_data_headers is not None and n_triggers_cleaned and n_dh_cleaned and trigger_idx[0] < fe_dh_idx[0]:
        return True, last_event_data_headers_cnt, n_triggers, n_dh  # FIXME: 0?
    # check for two consecutive triggers
    elif consecutive_triggers is None and prepend_data_headers == 0 and n_triggers_cleaned and n_dh_cleaned and trigger_idx[0] < fe_dh_idx[0]:
        return True, last_event_data_headers_cnt, n_triggers, n_dh  # FIXME: 0?
    elif prepend_data_headers is not None:
        trigger_idx += (prepend_data_headers + 1)
        fe_dh_idx += (prepend_data_headers + 1)
        # for histogramming add trigger at index 0
        trigger_idx = np.r_[0, trigger_idx]
        fe_dh_idx = np.r_[range(1, prepend_data_headers + 1), fe_dh_idx]

    event_hist, bins = np.histogram(fe_dh_idx, trigger_idx)
    if consecutive_triggers is None and np.any(event_hist == 0):
        return True, last_event_data_headers_cnt, n_triggers, n_dh
    elif consecutive_triggers and np.any(event_hist != consecutive_triggers):
        return True, last_event_data_headers_cnt, n_triggers, n_dh

    return False, last_event_data_headers_cnt, n_triggers, n_dh
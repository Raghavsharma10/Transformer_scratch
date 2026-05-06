def get_user_history (history_id=None):
    """
       Get all visible dataset infos of user history.
       Return a list of dict of each dataset.
    """ 
    history_id = history_id or os.environ['HISTORY_ID']
    gi = get_galaxy_connection(history_id=history_id, obj=False)
    hc = HistoryClient(gi)
    history = hc.show_history(history_id, visible=True, contents=True)
    return history
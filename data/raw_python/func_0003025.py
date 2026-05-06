def get_entry(journal_location, date):
    """
    args
    date - date object
    returns entry text or None if entry doesn't exist
    """
    if not isinstance(date, datetime.date):
        return None
    try:
        with open(build_journal_path(journal_location, date), "r") as entry_file:
            return entry_file.read()
    except IOError:
        return None
def record_entries(journal_location, entries):
    """
    args
    entry - list of entries to record
    """
    check_journal_dest(journal_location)
    current_date = datetime.datetime.today()
    date_header = current_date.strftime("%a %H:%M:%S %Y-%m-%d") + "\n"
    with open(build_journal_path(journal_location, current_date), "a") as date_file:
        entry_output = date_header
        # old style
        # for entry in entries:
        #     entry_output += "-" + entry + "\n"

        # new style
        entry_output += '-' + ' '.join(entries) + "\n"
        entry_output += "\n"
        date_file.write(entry_output)
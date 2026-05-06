def delete(bad_entry):
    """ Removes an entry from rc file. """
    entries = read()
    kept_entries = [x for x in entries if x.rstrip() != bad_entry]
    write(kept_entries)
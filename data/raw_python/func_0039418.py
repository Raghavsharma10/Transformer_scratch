def lookup_deleted_folder(event, filesystem, journal):
    """Lookup the parent folder in the journal content."""
    folder_events = (e for e in journal[event.parent_inode]
                     if 'DIRECTORY' in e.attributes
                     and 'FILE_DELETE' in e.changes)

    for folder_event in folder_events:
        path = lookup_deleted_folder(folder_event, filesystem, journal)

        return ntpath.join(path, event.name)

    return lookup_folder(event, filesystem)
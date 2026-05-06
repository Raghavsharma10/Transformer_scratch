def generate_timeline(usnjrnl, filesystem_content):
    """Aggregates the data collected from the USN journal
    and the filesystem content.

    """
    journal_content = defaultdict(list)
    for event in usnjrnl:
        journal_content[event.inode].append(event)

    for event in usnjrnl:
        try:
            dirent = lookup_dirent(event, filesystem_content, journal_content)

            yield UsnJrnlEvent(
                dirent.inode, dirent.path, dirent.size, dirent.allocated,
                event.timestamp, event.changes, event.attributes)
        except LookupError as error:
            LOGGER.debug(error)
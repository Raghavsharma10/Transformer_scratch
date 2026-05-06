def uncomment_or_update_or_append_line(filename, prefix, new_line, comment='#',
                                       keep_backup=True,
                                       update_or_append_line=update_or_append_line):
    '''Remove the comment of an commented out line and make the line "active".

    If such an commented out line not exists it would be appended.
    '''
    uncommented = update_or_append_line(filename, prefix=comment+prefix,
                                        new_line=new_line,
                                        keep_backup=keep_backup, append=False)
    if not uncommented:
        update_or_append_line(filename, prefix, new_line,
                              keep_backup=keep_backup, append=True)
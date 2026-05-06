def comment_out_line(filename, line, comment='#',
                     update_or_append_line=update_or_append_line):
    '''Comment line out by putting a comment sign in front of the line.

    If the file does not contain the line, the files content will not be
    changed (but the file will be touched in every case).
    '''
    update_or_append_line(filename, prefix=line, new_line=comment+line,
                          append=False)
def path_to_ls(fn):
    """ Converts an absolute path to an entry resembling the output of
        the ls command on most UNIX systems."""
    st = os.stat(fn)
    full_mode = 'rwxrwxrwx'
    mode = ''
    file_time = ''
    d = ''
    for i in range(9):
        # Incrementally builds up the 9 character string, using characters from the
        # fullmode (defined above) and mode bits from the stat() system call.
        mode += ((st.st_mode >> (8 - i)) & 1) and full_mode[i] or '-'
        d = (os.path.isdir(fn)) and 'd' or '-'
        file_time = time.strftime(' %b %d %H:%M ', time.gmtime(st.st_mtime))
    list_format = '{0}{1} 1 ftp ftp {2}\t{3}{4}'.format(d, mode, str(st.st_size), file_time, os.path.basename(fn))
    return list_format
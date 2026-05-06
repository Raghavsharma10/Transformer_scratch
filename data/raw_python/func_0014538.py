def set_file_attr(filename, attr):
        """
        Change a file's attributes on the local filesystem.  The contents of
        C{attr} are used to change the permissions, owner, group ownership,
        and/or modification & access time of the file, depending on which
        attributes are present in C{attr}.

        This is meant to be a handy helper function for translating SFTP file
        requests into local file operations.
        
        @param filename: name of the file to alter (should usually be an
            absolute path).
        @type filename: str
        @param attr: attributes to change.
        @type attr: L{SFTPAttributes}
        """
        if sys.platform != 'win32':
            # mode operations are meaningless on win32
            if attr._flags & attr.FLAG_PERMISSIONS:
                os.chmod(filename, attr.st_mode)
            if attr._flags & attr.FLAG_UIDGID:
                os.chown(filename, attr.st_uid, attr.st_gid)
        if attr._flags & attr.FLAG_AMTIME:
            os.utime(filename, (attr.st_atime, attr.st_mtime))
        if attr._flags & attr.FLAG_SIZE:
            open(filename, 'w+').truncate(attr.st_size)
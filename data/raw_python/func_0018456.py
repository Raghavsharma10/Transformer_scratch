def node_check_for_upload_create(self, relative_path, f):
        """Check if the given directory tree node has to be uploaded/created on the remote folder."""
        if not relative_path:
            # we're at the root of the shared directory tree
            relative_path = str()

        # the (absolute) local address of f.
        local_path = path_join(self.local_path, relative_path, f)
        try:
            l_st = os.lstat(local_path)
        except OSError as e:
            """A little background here.
            Sometimes, in big clusters configurations (mail, etc.),
            files could disappear or be moved, suddenly.
            There's nothing to do about it,
            system should be stopped before doing backups.
            Anyway, we log it, and skip it.
            """
            self.logger.error("error while checking {}: {}".format(relative_path, e))
            return

        if local_path in self.exclude_list:
            self.logger.info("Skipping excluded file %s.", local_path)
            return

        # the (absolute) remote address of f.
        remote_path = path_join(self.remote_path, relative_path, f)

        # First case: f is a directory
        if S_ISDIR(l_st.st_mode):
            # we check if the folder exists on the remote side
            # it has to be a folder, otherwise it would have already been
            # deleted
            try:
                self.sftp.stat(remote_path)
            except IOError:  # it doesn't exist yet on remote side
                self.sftp.mkdir(remote_path)

            self._match_modes(remote_path, l_st)

            # now, we should traverse f too (recursion magic!)
            self.check_for_upload_create(path_join(relative_path, f))

        # Second case: f is a symbolic link
        elif S_ISLNK(l_st.st_mode):
            # read the local link
            local_link = os.readlink(local_path)
            absolute_local_link = os.path.realpath(local_link)

            # is it absolute?
            is_absolute = local_link.startswith("/")
            # and does it point inside the shared directory?
            # add trailing slash (security)
            trailing_local_path = path_join(self.local_path, '')
            relpath = os.path.commonprefix(
                [absolute_local_link,
                 trailing_local_path]
            ) == trailing_local_path

            if relpath:
                relative_link = absolute_local_link[len(trailing_local_path):]
            else:
                relative_link = None

            """
            # Refactor them all, be efficient!

            # Case A: absolute link pointing outside shared directory
            #   (we can only update the remote part)
            if is_absolute and not relpath:
                self.create_update_symlink(local_link, remote_path)

            # Case B: absolute link pointing inside shared directory
            #   (we can leave it as it is or fix the prefix to match the one of the remote server)
            elif is_absolute and relpath:
                if self.fix_symlinks:
                    self.create_update_symlink(
                        join(
                            self.remote_path,
                            relative_link,
                        ),
                        remote_path
                    )
                else:
                    self.create_update_symlink(local_link, remote_path)

            # Case C: relative link pointing outside shared directory
            #   (all we can do is try to make the link anyway)
            elif not is_absolute and not relpath:
                self.create_update_symlink(local_link, remote_path)

            # Case D: relative link pointing inside shared directory
            #   (we preserve the relativity and link it!)
            elif not is_absolute and relpath:
                self.create_update_symlink(local_link, remote_path)
            """

            if is_absolute and relpath:
                if self.fix_symlinks:
                    self.create_update_symlink(
                        path_join(
                            self.remote_path,
                            relative_link,
                        ),
                        remote_path
                    )
            else:
                self.create_update_symlink(local_link, remote_path)

        # Third case: regular file
        elif S_ISREG(l_st.st_mode):
            try:
                r_st = self.sftp.lstat(remote_path)
                if self._file_need_upload(l_st, r_st):
                    self.file_upload(local_path, remote_path, l_st)
            except IOError as e:
                if e.errno == errno.ENOENT:
                    self.file_upload(local_path, remote_path, l_st)

        # Anything else.
        else:
            self.logger.warning("Skipping unsupported file %s.", local_path)
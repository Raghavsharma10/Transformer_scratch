def load_url(self,
                 url,
                 fname,
                 repo=None,
                 timeout=120,
                 post=None,
                 fail=False,
                 write=True,
                 json_sort=None,
                 cache_only=False,
                 archived_mode=None,
                 archived_task=None,
                 update_mode=None,
                 verify=False):
        """Load the given URL, or a cached-version.

        Load page from url or cached file, depending on the current settings.
        'archived' mode applies when `args.archived` is true (from
        `--archived` CL argument), and when this task has `Task.archived`
        also set to True.

        'archived' mode:
            * Try to load from cached file.
            * If cache does not exist, try to load from web.
            * If neither works, raise an error if ``fail == True``,
              otherwise return None
        non-'archived' mode:
            * Try to load from url, save to cache file.
            * If url fails, try to load existing cache file.
            * If neither works, raise an error if ``fail == True``,
              otherwise return None

        'update' mode:
            * In update mode, try to compare URL to cached file.
            * If URL fails, return None
              (cannot update)
            * If URL data matches cached data, return None
              (dont need to update)
            * If URL is different from data, return url data
              (proceed with update)

        Arguments
        ---------
        self
        url : str
            URL to download.
        fname : str
            Filename to which to save/load cached file.  Inludes suffix.
            NOTE: in general, this should be the source's BIBCODE.
        repo : str or None
            The full path of the data-repository the cached file should be
            saved/loaded from.  If 'None', then the current task is used to
            determine the repo.
        timeout : int
            Time (in seconds) after which a URL query should exit.
        post : dict
            List of arguments to post to URL when requesting it.
        archived : bool
            Load a previously archived version of the file.
        fail : bool
            If the file/url cannot be loaded, raise an error.
        write : bool
            Save a new copy of the cached file.
        json_sort : str or None
            If data is being saved to a json file, sort first by this str.
        quiet : bool
            Whether to emit error messages upon being unable to find files.
        verify : bool
            Whether to check for valid SSL cert when downloading

        """
        file_txt = None
        url_txt = None

        # Load default settings if needed
        # -------------------------------
        # Determine if we are running in archived mode
        if archived_mode is None:
            archived_mode = self.args.archived
        # Determine if this task is one which uses archived files
        if archived_task is None:
            archived_task = self.current_task.archived
        # Determine if running in update mode
        if update_mode is None:
            update_mode = self.args.update

        # Construct the cached filename
        if repo is None:
            repo = self.get_current_task_repo()
        cached_path = os.path.join(repo, fname)

        # Load cached file if it exists
        # ----------------------------
        if os.path.isfile(cached_path):
            with codecs.open(cached_path, 'r', encoding='utf8') as infile:
                file_txt = infile.read()
                self.log.debug("Task {}: Loaded from '{}'.".format(
                    self.current_task.name, cached_path))

        # In `archived` mode and task - try to return the cached page
        if archived_mode or (archived_task and not update_mode):
            if file_txt is not None:
                return file_txt

            # If this flag is set, don't even attempt to download from web
            if cache_only:
                return None

            # If file does not exist, log error, continue
            else:
                self.log.error("Task {}: Cached file '{}' does not exist.".
                               format(self.current_task.name, cached_path))

        # Load url.  'None' is returned on failure - handle that below
        url_txt = self.download_url(
            url, timeout, fail=False, post=post, verify=verify)

        # At this point, we might have both `url_txt` and `file_txt`
        # If either of them failed, then they are set to None

        # If URL download failed, error or return cached data
        # ---------------------------------------------------
        if url_txt is None:
            # Both sources failed
            if file_txt is None:
                err_str = "Both url and file retrieval failed!"
                # If we should raise errors on failure
                if fail:
                    err_str += " `fail` set."
                    self.log.error(err_str)
                    raise RuntimeError(err_str)
                # Otherwise warn and return None
                self.log.warning(err_str)
                return None

            # Otherwise, if only url failed, return file data
            else:
                # If we are trying to update, but the url failed, then return
                # None
                if update_mode:
                    self.log.error(
                        "Cannot check for updates, url download failed.")
                    return None
                # Otherwise, return file data
                self.log.warning("URL download failed, using cached data.")
                return file_txt

        # Here: `url_txt` exists, `file_txt` may exist or may be None
        # Determine if update should happen, and if file should be resaved

        # Write new url_txt to cache file
        # -------------------------------
        if write:
            self.log.info(
                "Writing `url_txt` to file '{}'.".format(cached_path))
            self._write_cache_file(url_txt, cached_path, json_sort=json_sort)
        # If `file_txt` doesnt exist but were not writing.. warn
        elif file_txt is None:
            err_str = "Warning: cached file '{}' does not exist.".format(
                cached_path)
            err_str += " And is not being saved."
            self.log.warning(err_str)

        # Check if we need to update this data
        # ------------------------------------
        # If both `url_txt` and `file_txt` exist and update mode check MD5
        if file_txt is not None and update_mode:
            from hashlib import md5
            url_md5 = md5(url_txt.encode('utf-8')).hexdigest()
            file_md5 = md5(file_txt.encode('utf-8')).hexdigest()
            self.log.debug("URL: '{}', File: '{}'.".format(url_md5, file_md5))
            # If the data is the same, no need to parse (update), return None
            if url_md5 == file_md5:
                self.log.info(
                    "Skipping file '{}', no changes.".format(cached_path))
                return None
            else:
                self.log.info("File '{}' has been updated".format(cached_path))
                # Warn if we didnt save a new copy
                if not write:
                    err_str = "Warning: updated data not saved to file."
                    self.log.warning(err_str)

        return url_txt
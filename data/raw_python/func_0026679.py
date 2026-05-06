def download_url(self, url, timeout, fail=False, post=None, verify=True):
        """Download text from the given url.

        Returns `None` on failure.

        Arguments
        ---------
        self
        url : str
            URL web address to download.
        timeout : int
            Duration after which URL request should terminate.
        fail : bool
            If `True`, then an error will be raised on failure.
            If `False`, then 'None' is returned on failure.
        post : dict
            List of arguments to post to URL when requesting it.
        verify : bool
            Whether to check for valid SSL cert when downloading

        Returns
        -------
        url_txt : str or None
            On success the text of the url is returned.  On failure `None` is
            returned.

        """
        _CODE_ERRORS = [500, 307, 404]
        import requests
        session = requests.Session()

        try:
            headers = {
                'User-Agent':
                'Mozilla/5.0 (Macintosh; Intel Mac OS X '
                '10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/39.0.2171.95 Safari/537.36'
            }
            if post:
                response = session.post(
                    url,
                    timeout=timeout,
                    headers=headers,
                    data=post,
                    verify=verify)
            else:
                response = session.get(
                    url, timeout=timeout, headers=headers, verify=verify)
            response.raise_for_status()
            # Look for errors
            for xx in response.history:
                xx.raise_for_status()
                if xx.status_code in _CODE_ERRORS:
                    self.log.error("URL response returned status code '{}'".
                                   format(xx.status_code))
                    raise

            url_txt = response.text
            self.log.debug("Task {}: Loaded `url_txt` from '{}'.".format(
                self.current_task.name, url))

        except (KeyboardInterrupt, SystemExit):
            raise

        except Exception as err:
            err_str = ("URL Download of '{}' failed ('{}')."
                       .format(url, str(err)))
            # Raise an error on failure
            if fail:
                err_str += " and `fail` is set."
                self.log.error(err_str)
                raise RuntimeError(err_str)
            # Log a warning on error, and return None
            else:
                self.log.warning(err_str)
                return None

        return url_txt
def urlopen(self, method, url, body=None, headers=None, retries=None,
                redirect=True, assert_same_host=True, timeout=_Default,
                pool_timeout=None, release_conn=None, **response_kw):
        """
        Get a connection from the pool and perform an HTTP request. This is the
        lowest level call for making a request, so you'll need to specify all
        the raw details.

        .. note::

           More commonly, it's appropriate to use a convenience method provided
           by :class:`.RequestMethods`, such as :meth:`request`.

        .. note::

           `release_conn` will only behave as expected if
           `preload_content=False` because we want to make
           `preload_content=False` the default behaviour someday soon without
           breaking backwards compatibility.

        :param method:
            HTTP request method (such as GET, POST, PUT, etc.)

        :param body:
            Data to send in the request body (useful for creating
            POST requests, see HTTPConnectionPool.post_url for
            more convenience).

        :param headers:
            Dictionary of custom headers to send, such as User-Agent,
            If-None-Match, etc. If None, pool headers are used. If provided,
            these headers completely replace any pool-specific headers.

        :param retries:
            Configure the number of retries to allow before raising a
            :class:`~urllib3.exceptions.MaxRetryError` exception.

            Pass ``None`` to retry until you receive a response. Pass a
            :class:`~urllib3.util.retry.Retry` object for fine-grained control
            over different types of retries.
            Pass an integer number to retry connection errors that many times,
            but no other types of errors. Pass zero to never retry.

            If ``False``, then retries are disabled and any exception is raised
            immediately. Also, instead of raising a MaxRetryError on redirects,
            the redirect response will be returned.

        :type retries: :class:`~urllib3.util.retry.Retry`, False, or an int.

        :param redirect:
            If True, automatically handle redirects (status codes 301, 302,
            303, 307, 308). Each redirect counts as a retry. Disabling retries
            will disable redirect, too.

        :param assert_same_host:
            If ``True``, will make sure that the host of the pool requests is
            consistent else will raise HostChangedError. When False, you can
            use the pool on an HTTP proxy and request foreign hosts.

        :param timeout:
            If specified, overrides the default timeout for this one
            request. It may be a float (in seconds) or an instance of
            :class:`urllib3.util.Timeout`.

        :param pool_timeout:
            If set and the pool is set to block=True, then this method will
            block for ``pool_timeout`` seconds and raise EmptyPoolError if no
            connection is available within the time period.

        :param release_conn:
            If False, then the urlopen call will not release the connection
            back into the pool once a response is received (but will release if
            you read the entire contents of the response such as when
            `preload_content=True`). This is useful if you're not preloading
            the response's content immediately. You will need to call
            ``r.release_conn()`` on the response ``r`` to return the connection
            back into the pool. If None, it takes the value of
            ``response_kw.get('preload_content', True)``.

        :param \**response_kw:
            Additional parameters are passed to
            :meth:`urllib3.response.HTTPResponse.from_httplib`
        """
        if headers is None:
            headers = self.headers

        if not isinstance(retries, Retry):
            retries = Retry.from_int(retries, redirect=redirect, default=self.retries)

        if release_conn is None:
            release_conn = response_kw.get('preload_content', True)

        # Check host
        if assert_same_host and not self.is_same_host(url):
            raise HostChangedError(self, url, retries)

        conn = None

        # Merge the proxy headers. Only do this in HTTP. We have to copy the
        # headers dict so we can safely change it without those changes being
        # reflected in anyone else's copy.
        if self.scheme == 'http':
            headers = headers.copy()
            headers.update(self.proxy_headers)

        # Must keep the exception bound to a separate variable or else Python 3
        # complains about UnboundLocalError.
        err = None

        try:
            # Request a connection from the queue.
            timeout_obj = self._get_timeout(timeout)
            conn = self._get_conn(timeout=pool_timeout)

            conn.timeout = timeout_obj.connect_timeout

            is_new_proxy_conn = self.proxy is not None and not getattr(conn, 'sock', None)
            if is_new_proxy_conn:
                self._prepare_proxy(conn)

            # Make the request on the httplib connection object.
            httplib_response = self._make_request(conn, method, url,
                                                  timeout=timeout_obj,
                                                  body=body, headers=headers)

            # If we're going to release the connection in ``finally:``, then
            # the request doesn't need to know about the connection. Otherwise
            # it will also try to release it and we'll have a double-release
            # mess.
            response_conn = not release_conn and conn

            # Import httplib's response into our own wrapper object
            response = HTTPResponse.from_httplib(httplib_response,
                                                 pool=self,
                                                 connection=response_conn,
                                                 **response_kw)

            # else:
            #     The connection will be put back into the pool when
            #     ``response.release_conn()`` is called (implicitly by
            #     ``response.read()``)

        except Empty:
            # Timed out by queue.
            raise EmptyPoolError(self, "No pool connections are available.")

        except (BaseSSLError, CertificateError) as e:
            # Close the connection. If a connection is reused on which there
            # was a Certificate error, the next request will certainly raise
            # another Certificate error.
            if conn:
                conn.close()
                conn = None
            raise SSLError(e)

        except SSLError:
            # Treat SSLError separately from BaseSSLError to preserve
            # traceback.
            if conn:
                conn.close()
                conn = None
            raise

        except (TimeoutError, HTTPException, SocketError, ConnectionError) as e:
            if conn:
                # Discard the connection for these exceptions. It will be
                # be replaced during the next _get_conn() call.
                conn.close()
                conn = None

            if isinstance(e, SocketError) and self.proxy:
                e = ProxyError('Cannot connect to proxy.', e)
            elif isinstance(e, (SocketError, HTTPException)):
                e = ProtocolError('Connection aborted.', e)

            retries = retries.increment(method, url, error=e, _pool=self,
                                        _stacktrace=sys.exc_info()[2])
            retries.sleep()

            # Keep track of the error for the retry warning.
            err = e

        finally:
            if release_conn:
                # Put the connection back to be reused. If the connection is
                # expired then it will be None, which will get replaced with a
                # fresh connection during _get_conn.
                self._put_conn(conn)

        if not conn:
            # Try again
            log.warning("Retrying (%r) after connection "
                        "broken by '%r': %s" % (retries, err, url))
            return self.urlopen(method, url, body, headers, retries,
                                redirect, assert_same_host,
                                timeout=timeout, pool_timeout=pool_timeout,
                                release_conn=release_conn, **response_kw)

        # Handle redirect?
        redirect_location = redirect and response.get_redirect_location()
        if redirect_location:
            if response.status == 303:
                method = 'GET'

            try:
                retries = retries.increment(method, url, response=response, _pool=self)
            except MaxRetryError:
                if retries.raise_on_redirect:
                    raise
                return response

            log.info("Redirecting %s -> %s" % (url, redirect_location))
            return self.urlopen(method, redirect_location, body, headers,
                    retries=retries, redirect=redirect,
                    assert_same_host=assert_same_host,
                    timeout=timeout, pool_timeout=pool_timeout,
                    release_conn=release_conn, **response_kw)

        # Check if we should retry the HTTP response.
        if retries.is_forced_retry(method, status_code=response.status):
            retries = retries.increment(method, url, response=response, _pool=self)
            retries.sleep()
            log.info("Forced retry: %s" % url)
            return self.urlopen(method, url, body, headers,
                    retries=retries, redirect=redirect,
                    assert_same_host=assert_same_host,
                    timeout=timeout, pool_timeout=pool_timeout,
                    release_conn=release_conn, **response_kw)

        return response
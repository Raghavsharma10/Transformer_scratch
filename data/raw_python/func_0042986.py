def set_more_headers(self, req, extra_headers=None):
        """Set content-type, content-md5, date to the request
        Returns a new `PreparedRequest`

        :param req: the origin unsigned request
        :param extra_headers: extra headers you want to set, pass as dict
        """
        oss_url = url.URL(req.url)
        req.headers.update(extra_headers or {})

        # set content-type
        content_type = req.headers.get("content-type")
        if content_type is None:
            content_type, __ = mimetypes.guess_type(oss_url.path)
        req.headers["content-type"] = content_type or self.DEFAULT_TYPE
        logger.info("set content-type to: {0}".format(content_type))

        # set date
        if self._expires is None:
            req.headers.setdefault(
                "date",
                time.strftime(self.TIME_FMT, time.gmtime())
            )
        else:
            req.headers["content-type"] = ""
            req.headers["date"] = self._expires

        logger.info("set date to: {0}".format(req.headers["date"]))

        # set content-md5
        if req.body is None:
            content_md5 = ""
        else:
            content_md5 = req.headers.get("content-md5", "")
            if not content_md5 and self._allow_empty_md5 is False:
                content_md5 = utils.cal_b64md5(req.body)
        req.headers["content-md5"] = content_md5
        logger.info("content-md5 to: [{0}]".format(content_md5))

        return req
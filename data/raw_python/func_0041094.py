def _str2datetime(self, datetimestr):
        """Parse datetime from string. If no template matches this string,
        raise Error. Please go
        https://github.com/MacHu-GWU/rolex-project/issues
        submit your datetime string. I 'll update templates asap.

        This method is faster than :meth:`dateutil.parser.parse`.

        :param datetimestr: a string represent a datetime
        :type datetimestr: str
        :return: a datetime object

        **中文文档**

        从string解析datetime。首先尝试默认模板, 如果失败了, 则尝试所有的模板。
        一旦尝试成功, 就将当前成功的模板保存为默认模板。这样做在当你待解析的
        字符串非常多, 且模式单一时, 只有第一次尝试耗时较多, 之后就非常快了。
        该方法要快过 :meth:`dateutil.parser.parse` 方法。
        """
        if datetimestr is None:
            raise ValueError(
                "Parser must be a string or character stream, not NoneType")

        # try default datetime template
        try:
            return datetime.strptime(
                datetimestr, self.default_datetime_template)
        except:
            pass

        # try every datetime templates
        for template in DatetimeTemplates:
            try:
                dt = datetime.strptime(datetimestr, template)
                self.default_datetime_template = template
                return dt
            except:
                pass

        # raise error
        dt = parser.parse(datetimestr)
        self.str2datetime = parser.parse

        return dt
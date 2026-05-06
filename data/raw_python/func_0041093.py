def str2date(self, datestr):
        """Parse date from string. If no template matches this string,
        raise Error. Please go
        https://github.com/MacHu-GWU/rolex-project/issues
        submit your date string. I 'll update templates asap.

        This method is faster than :meth:`dateutil.parser.parse`.

        :param datestr: a string represent a date
        :type datestr: str
        :return: a date object

        **中文文档**

        从string解析date。首先尝试默认模板, 如果失败了, 则尝试所有的模板。
        一旦尝试成功, 就将当前成功的模板保存为默认模板。这样做在当你待解析的
        字符串非常多, 且模式单一时, 只有第一次尝试耗时较多, 之后就非常快了。
        该方法要快过 :meth:`dateutil.parser.parse` 方法。
        """
        if datestr is None:
            raise ValueError(
                "Parser must be a string or character stream, not NoneType")

        # try default date template
        try:
            return datetime.strptime(
                datestr, self.default_date_template).date()
        except:
            pass

        # try every datetime templates
        for template in DateTemplates:
            try:
                dt = datetime.strptime(datestr, template)
                self.default_date_template = template
                return dt.date()
            except:
                pass

        # raise error
        raise Exception("Unable to parse date from: %r" % datestr)
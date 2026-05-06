def log(obj1, obj2, sym, cname=None, aname=None, result=None):  # pylint: disable=R0913
        """Log the objects being compared and the result.

        When no result object is specified, subsequence calls will have an
        increased indentation level. The indentation level is decreased
        once a result object is provided.

        @param obj1: first object
        @param obj2: second object
        @param sym: operation being performed ('==' or '%')
        @param cname: name of class (when attributes are being compared)
        @param aname: name of attribute (when attributes are being compared)
        @param result: outcome of comparison

        """
        fmt = "{o1} {sym} {o2} : {r}"
        if cname or aname:
            assert cname and aname  # both must be specified
            fmt = "{c}.{a}: " + fmt

        if result is None:
            result = '...'
            fmt = _Indent.indent(fmt)
            _Indent.more()
        else:
            _Indent.less()
            fmt = _Indent.indent(fmt)

        msg = fmt.format(o1=repr(obj1), o2=repr(obj2),
                         c=cname, a=aname, sym=sym, r=result)
        logging.info(msg)
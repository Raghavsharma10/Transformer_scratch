def tablename_from_link(klass, link):
        """
        Helper method for URL's that look like /api/now/v1/table/FOO/sys_id etc.
        """
        arr = link.split("/")
        i = arr.index("table")
        tn = arr[i+1]
        return tn
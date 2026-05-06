def startswith(text, ignore_case=True):
        """
        Test if a string-field start with ``text``.

        Example::

            filters = {"path": Text.startswith(r"C:\\")}
        """
        if ignore_case:
            compiled = re.compile(
                "^%s" % text.replace("\\", "\\\\"), re.IGNORECASE)
        else:  # pragma: no cover
            compiled = re.compile("^%s" % text.replace("\\", "\\\\"))

        return {"$regex": compiled}
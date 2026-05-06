def fileinfo(fileobj, filename=None, content_type=None, existing=None):
        """Tries to extract from the given input the actual file object, filename and content_type

        This is used by the create and replace methods to correctly deduce their parameters
        from the available information when possible.
        """
        return _FileInfo(fileobj, filename, content_type).get_info(existing)
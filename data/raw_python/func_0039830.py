def resolve_local(self, uri, base_uri, ref):
        """
        Resolve a local ``uri``.
         Does not check the store first.

         :argument str uri: the URI to resolve
         :returns: the retrieved document

        """
        # read it from the filesystem
        file_path = None
        # make the reference saleskingstyle
        item_name = None
        if (uri.startswith(u"file") or
                uri.startswith(u"File")):
            if ref.startswith(u"./"):
                ref = ref.split(u"./")[-1]
                org_ref = ref
            if ref.find(u"#properties") != -1:
                ref = ref.split(u"#properties")[0]
            if ref.find(u".json") != -1:
                item_name = ref.split(u".json")[0]

        # on windwos systesm this needs to happen
        if base_uri.startswith(u"file://") is True:
            base_uri = base_uri.split(u"file://")[1]
        elif base_uri.startswith(u"File://") is True:
            base_uri = base_uri.split(u"File://")[1]

        file_path = os.path.join(base_uri, ref)
        result = None
        try:
            schema_file = open(file_path, "r").read()
            result = json.loads(schema_file.decode("utf-8"))
        except IOError as e:
            log.error(u"file not found %s" % e)
            msg = "Could not find schema file. %s" % file_path
            raise SalesKingException("SCHEMA_NOT_FOUND", msg)

        if self.cache_remote:
            self.store[uri] = result
        return result
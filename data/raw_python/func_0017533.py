def flatten_urlinfo(urlinfo, shorter_keys=True):
    """ Takes a urlinfo object and returns a flat dictionary."""
    def flatten(value, prefix=""):
        if is_string(value):
            _result[prefix[1:]] = value
            return
        try:
            len(value)
        except (AttributeError, TypeError):  # a leaf
            _result[prefix[1:]] = value
            return

        try:
            items = value.items()
        except AttributeError:  # an iterable, but not a dict
            last_prefix = prefix.split(".")[-1]
            if shorter_keys:
                prefix = "." + last_prefix

            if last_prefix == "Country":
                for v in value:
                    country = v.pop("@Code")
                    flatten(v, ".".join([prefix, country]))
            elif last_prefix in ["RelatedLink", "CategoryData"]:
                for i, v in enumerate(value):
                    flatten(v, ".".join([prefix, str(i)]))
            elif value[0].get("TimeRange"):
                for v in value:
                    time_range = ".".join(tuple(v.pop("TimeRange").items())[0])
                    # python 3 odict_items don't support indexing
                    if v.get("DataUrl"):
                        time_range = ".".join([v.pop("DataUrl"), time_range])
                    flatten(v, ".".join([prefix, time_range]))
            else:
                msg = prefix + " contains a list we don't know how to flatten."
                raise NotImplementedError(msg)
        else:  # a dict, go one level deeper
            for k, v in items:
                flatten(v, ".".join([prefix, k]))

    _result = {}
    info = xmltodict.parse(str(urlinfo))
    flatten(info["aws:UrlInfoResponse"]["Response"]["UrlInfoResult"]["Alexa"])
    _result["OutputTimestamp"] = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    return _result
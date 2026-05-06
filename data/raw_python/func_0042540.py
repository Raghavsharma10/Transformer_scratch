def keys(self, prefix=None, delimiter=None):
        """
        :param prefix:  NOT A STRING PREFIX, RATHER PATH ID PREFIX (MUST MATCH TO NEXT "." OR ":")
        :param delimiter:  TO GET Prefix OBJECTS, RATHER THAN WHOLE KEYS
        :return: SET OF KEYS IN BUCKET, OR
        """
        if delimiter:
            # WE REALLY DO NOT GET KEYS, BUT RATHER Prefix OBJECTS
            # AT LEAST THEY ARE UNIQUE
            candidates = [k.name.rstrip(delimiter) for k in self.bucket.list(prefix=prefix, delimiter=delimiter)]
        else:
            candidates = [strip_extension(k.key) for k in self.bucket.list(prefix=prefix)]

        if prefix == None:
            return set(c for c in candidates if c != "0.json")
        else:
            return set(k for k in candidates if k == prefix or k.startswith(prefix + ".") or k.startswith(prefix + ":"))
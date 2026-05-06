def match(Class, path, pattern, flags=re.I, sortkey=None, ext=None):
        """for a given path and regexp pattern, return the files that match"""
        return sorted(
            [
                Class(fn=fn)
                for fn in rglob(path, f"*{ext or ''}")
                if re.search(pattern, os.path.basename(fn), flags=flags) is not None
                and os.path.basename(fn)[0] != '~'  # omit temp files
            ],
            key=sortkey,
        )
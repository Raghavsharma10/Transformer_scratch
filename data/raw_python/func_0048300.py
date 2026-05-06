def normalise_filled(self, meta, val):
        """Only care about valid image names"""
        available = list(meta.everything["images"].keys())
        val = sb.formatted(sb.string_spec(), formatter=MergedOptionStringFormatter).normalise(meta, val)
        if val not in available:
            raise BadConfiguration("Specified image doesn't exist", specified=val, available=available)
        return val
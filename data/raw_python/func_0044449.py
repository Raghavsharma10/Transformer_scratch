def _filter_image(self, url):
        "The param is the image URL, which is returned if it passes all the filters."
        return reduce(lambda f, g: f and g(f), 
        [
            filters.AdblockURLFilter()(url),
            filters.NoImageFilter(),
            filters.SizeImageFilter(),
            filters.MonoImageFilter(),
            filters.FormatImageFilter(),
        ])
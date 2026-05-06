def _load(self, titles=[], descriptions=[], images=[], urls=[], **kwargs):
        """
        Loads extracted data into Summary.
        Performs validation and filtering on-the-fly, and sets the
        non-plural fields to the best specific item so far.
        If GET_ALL_DATA is False, it gets only the first valid item.
        """
        enough = lambda items: items # len(items) >= MAX_ITEMS

        if config.GET_ALL_DATA or not enough(self.titles):
            titles = filter(None, map(self._clean_text, titles))
            self.titles.extend(titles)

        if config.GET_ALL_DATA or not enough(self.descriptions):
            descriptions = filter(None, map(self._clean_text, descriptions))
            self.descriptions.extend(descriptions)

        ## Never mind the urls, they can be bad not worth it
        # if config.GET_ALL_DATA or not enough(self.urls):
            # # urls = [self._clean_url(u) for u in urls]
            # urls = filter(None, map(self._clean_url, urls))
            # self.urls.extend(urls)

        if config.GET_ALL_DATA:
            # images = [i for i in [self._filter_image(i) for i in images] if i]
            images = filter(None, map(self._filter_image, images))
            self.images.extend(images)
        elif not enough(self.images):
            for i in images:
                image = self._filter_image(i)
                if image:
                    self.images.append(image)
                if enough(self.images):
                    break
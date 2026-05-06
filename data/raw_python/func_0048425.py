def _link_to_img(self):
        """
        Generates a link to the user's Gravatar.
        
        >>> Gravatar('gridaphobe@gmail.com')._link_to_img()
        'http://www.gravatar.com/avatar/16b87da510d278999c892cdbdd55c1b6?s=80&r=g'
        """
        # make sure options are valid
        if self.rating.lower() not in RATINGS:
            raise InvalidRatingError(self.rating)
        if not (MIN_SIZE <= self.size <= MAX_SIZE):
            raise InvalidSizeError(self.size)
        
        url = ''
        if self.secure:
            url = SECURE_BASE_URL
        else:
            url = BASE_URL

        options = {'s' : self.size, 'r' : self.rating}
        if self.default is not None:
            options['d'] = self.default
        url += self.hash + '?' + urlencode(options)
        return url
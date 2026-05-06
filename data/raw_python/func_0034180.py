def handleMatch(self, m):
        """
        Handles user input into [magic] tag, processes it,
        and inserts the returned URL into an <img> tag
        through a Python ElementTree <img> Element.
        """
        userStr = m.group(3)
        # print(userStr)
        imgURL = processString(userStr)
        # print(imgURL)
        el = etree.Element('img')
        # Sets imgURL to 'src' attribute of <img> tag element
        el.set('src', imgURL)       
        el.set('alt', userStr)
        el.set('title', userStr)
        return el
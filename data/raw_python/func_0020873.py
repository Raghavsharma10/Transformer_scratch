def push_content(self, title, url,
                     images=None, date=None, expire_date=None,
                     description=None, location=None, price=None,
                     tags=None,
                     author=None, site_name=None,
                     spider=None, vars=None):

        """
        Push a new piece of content to Sailthru.

        Expected names for the `images` argument's map are "full" and "thumb"
        Expected format for `location` should be [longitude,latitude]

        @param title: title string for the content
        @param url: URL string for the content
        @param images: map of image names
        @param date: date string
        @param expire_date: date string for when the content expires
        @param description: description for the content
        @param location: location of the content
        @param price: price for the content
        @param tags: list or comma separated string values
        @param author: author for the content
        @param site_name: site name for the content
        @param spider: truthy value to force respidering content
        @param vars: replaceable vars dictionary

        """
        vars = vars or {}
        data = {'title': title,
                'url': url}
        if images is not None:
            data['images'] = images
        if date is not None:
            data['date'] = date
        if expire_date is not None:
            data['expire_date'] = date
        if location is not None:
            data['location'] = date
        if price is not None:
            data['price'] = price
        if description is not None:
            data['description'] = description
        if site_name is not None:
            data['site_name'] = images
        if author is not None:
            data['author'] = author
        if spider:
            data['spider'] = 1
        if tags is not None:
            data['tags'] = ",".join(tags) if isinstance(tags, list) else tags
        if len(vars) > 0:
            data['vars'] = vars.copy()
        return self.api_post('content', data)
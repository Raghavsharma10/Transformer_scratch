def item_extra_kwargs(self, item):
        """
        Returns an extra keyword arguments dictionary that is used with
        the 'add_item' call of the feed generator.
        Add the fields of the item, to be used by the custom feed generator.
        """
        if use_feed_image:
            feed_image = item.feed_image
            if feed_image:
                image_complete_url = urljoin(
                    self.get_site_url(), feed_image.file.url
                )
            else:
                image_complete_url = ""

        content_field = getattr(item, self.item_content_field)
        try:
            content = expand_db_html(content_field)
        except:
            content = content_field.__html__()

        soup = BeautifulSoup(content, 'html.parser')
        # Remove style attribute to remove large botton padding
        for div in soup.find_all("div", {'class': 'responsive-object'}):
            del div['style']
        # Add site url to image source
        for img_tag in soup.findAll('img'):
            if img_tag.has_attr('src'):
                img_tag['src'] = urljoin(self.get_site_url(), img_tag['src'])

        fields_to_add = {
            'content': soup.prettify(formatter="html"),
        }

        if use_feed_image:
            fields_to_add['image'] = image_complete_url
        else:
            fields_to_add['image'] = ""

        return fields_to_add
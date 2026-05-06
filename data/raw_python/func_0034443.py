def get_show_name(self):
        """
        Get video show name from the website. It's located in the div with 'data-hover'
        attribute under the 'title' key.

        Returns:
            str: Video show name.

        """
        div = self.soup.find('div', attrs={'data-hover': True})
        data = json.loads(div['data-hover'])
        show_name = data.get('title')

        return show_name
def fetch_new_release_category(self, category_id, terr=KKBOXTerritory.TAIWAN):
        '''
        Fetches new release categories by given ID.

        :param category_id: the station ID.
        :type category_id: str
        :param terr: the current territory.
        :return: API response.
        :rtype: list

        See `https://docs-en.kkbox.codes/v1.1/reference#newreleasecategories-category_id`
        '''
        url = 'https://api.kkbox.com/v1.1/new-release-categories/%s' % category_id
        url += '?' + url_parse.urlencode({'territory': terr})
        return self.http._post_data(url, None, self.http._headers_with_access_token())
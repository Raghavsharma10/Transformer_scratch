def _get_accept_languages_in_order(self):
        """
        Reads an Accept HTTP header and returns an array of Media Type string in descending weighted order

        :return: List of URIs of accept profiles in descending request order
        :rtype: list
        """
        try:
            # split the header into individual URIs, with weights still attached
            profiles = self.request.headers['Accept-Language'].split(',')
            # remove \s
            profiles = [x.replace(' ', '').strip() for x in profiles]

            # split off any weights and sort by them with default weight = 1
            profiles = [(float(x.split(';')[1].replace('q=', '')) if len(x.split(';')) == 2 else 1, x.split(';')[0]) for x in profiles]

            # sort profiles by weight, heaviest first
            profiles.sort(reverse=True)

            return[x[1] for x in profiles]
        except Exception as e:
            raise ViewsFormatsException(
                'You have requested a language using an Accept-Language header that is incorrectly formatted.')
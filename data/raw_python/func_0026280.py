def query(self, lat=None, lon=None, osm_id=None, osm_type=None,
              acceptlanguage='', zoom=18):
        """
        Issue a reverse geocoding query for a place given
        by *lat* and *lon*, or by *osm_id* and *osm_type*
        to the Nominatim instance and return the decoded results

        :param lat: the geograpical latitude of the place
        :param lon: the geograpical longitude of the place
        :param osm_id: openstreetmap identifier osm_id
        :type osm_id: str
        :param osm_type: openstreetmap type osm_type
        :type osm_type: str
        :param acceptlanguage: rfc2616 language code
        :type acceptlanguage: str or None
        :param zoom: zoom factor between from 0 to 18
        :type zoom: int or None or a key in :data:`zoom_aliases`
        :param countrycodes: restrict the search to countries
             given by their ISO 3166-1alpha2 codes (cf.
             https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2 )
        :type countrycodes: str iterable
        :returns: a list of search results (each a dict)
        :rtype: list or None
        :raise: NominatimException if invalid zoom value
        """
        url = self.url
        if osm_id is not None and osm_type not in ('N', 'W', 'R'):
            raise NominatimException('invalid osm_type')
        if osm_id is not None and osm_type is not None:
            url += '&osm_id=' + osm_id + '&osm_type=' + osm_type
        elif lat is not None and lon is not None:
            url += '&lat=' + str(lat) + '&lon=' + str(lon)
        else:
            return None
        if acceptlanguage:
            url += '&accept-language=' + acceptlanguage
        if zoom in zoom_aliases:
            zoom = zoom_aliases[zoom]
        if not isinstance(zoom, int) or zoom < 0 or zoom > 18:
            raise NominatimException('zoom must effectively be betwen 0 and 18')
        url +='&zoom=' + str(zoom)
        return self.request(url)
def find_times(self, site, frametype, gpsstart=None, gpsend=None):
        """Query the LDR for times for which frames are avaliable

        Use gpsstart and gpsend to restrict the returned times to
        this semiopen interval.

        @returns: L{segmentlist<pycbc_glue.segments.segmentlist>}

        @param site:
            single-character name of site to match
        @param frametype:
            name of frametype to match
        @param gpsstart:
            integer GPS start time of query
        @param gpsend:
            integer GPS end time of query

        @type       site: L{str}
        @type  frametype: L{str}
        @type   gpsstart: L{int}
        @type     gpsend: L{int}
        """
        if gpsstart and gpsend:
            url = ("%s/gwf/%s/%s/segments/%s,%s.json"
                   % (_url_prefix, site, frametype, gpsstart, gpsend))
        else:
            url = ("%s/gwf/%s/%s/segments.json"
                   % (_url_prefix, site, frametype))

        response = self._requestresponse("GET", url)
        segmentlist = decode(response.read())
        return segments.segmentlist(map(segments.segment, segmentlist))
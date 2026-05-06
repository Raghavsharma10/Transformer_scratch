def find_frame_urls(self, site, frametype, gpsstart, gpsend,
                        match=None, urltype=None, on_gaps="warn"):
        """Find the framefiles for the given type in the [start, end) interval
        frame

        @param site:
            single-character name of site to match
        @param frametype:
            name of frametype to match
        @param gpsstart:
            integer GPS start time of query
        @param gpsend:
            integer GPS end time of query
        @param match:
            regular expression to match against
        @param urltype:
            file scheme to search for (e.g. 'file')
        @param on_gaps:
            what to do when the requested frame isn't found, one of:
                - C{'warn'} (default): print a warning,
                - C{'error'}: raise an L{RuntimeError}, or
                - C{'ignore'}: do nothing

        @type       site: L{str}
        @type  frametype: L{str}
        @type   gpsstart: L{int}
        @type     gpsend: L{int}
        @type      match: L{str}
        @type    urltype: L{str}
        @type    on_gaps: L{str}

        @returns: L{Cache<pycbc_glue.lal.Cache>}

        @raises RuntimeError: if gaps are found and C{on_gaps='error'}
        """
        if on_gaps not in ("warn", "error", "ignore"):
            raise ValueError("on_gaps must be 'warn', 'error', or 'ignore'.")
        url = ("%s/gwf/%s/%s/%s,%s"
               % (_url_prefix, site, frametype, gpsstart, gpsend))
        # if a URL type is specified append it to the path
        if urltype:
            url += "/%s" % urltype
        # request JSON output
        url += ".json"
        # append a regex if input
        if match:
            url += "?match=%s" % match
        # make query
        response = self._requestresponse("GET", url)
        urllist  = decode(response.read())

        out = lal.Cache([lal.CacheEntry.from_T050017(x,
                         coltype=self.LIGOTimeGPSType) for x in urllist])

        if on_gaps == "ignore":
            return out
        else:
            span    = segments.segment(gpsstart, gpsend)
            seglist = segments.segmentlist(e.segment for e in out).coalesce()
            missing = (segments.segmentlist([span]) - seglist).coalesce()
            if span in seglist:
                return out
            else:
                msg = "Missing segments: \n%s" % "\n".join(map(str, missing))
                if on_gaps=="warn":
                    sys.stderr.write("%s\n" % msg)
                    return out
                else:
                    raise RuntimeError(msg)
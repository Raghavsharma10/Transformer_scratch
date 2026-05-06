def _get_subfolder(self, foldername, returntype,
                       params=None, file_data=None):
        """Return an object of the requested type with the path relative
           to the current object's URL. Optionally, query parameters
           may be set."""
        newurl = compat.urljoin(self.url, compat.quote(foldername), False)

        params = params or {}
        file_data = file_data or {}

        # Add the key-value pairs sent in params to query string if they
        # are so defined.
        query_dict = {}
        url_tuple = compat.urlsplit(newurl)
        urllist = list(url_tuple)

        if params:
            # As above, pull out first element from parse_qs' values
            query_dict = dict((k, v[0]) for k, v in 
                               cgi.parse_qs(urllist[3]).items())
            for key, val in params.items():
                # Lowercase bool string
                if isinstance(val, bool):
                    query_dict[key] = str(val).lower()
                # Special case: convert an envelope to .bbox in the bb
                # parameter
                elif isinstance(val, geometry.Envelope):
                    query_dict[key] = val.bbox
                # Another special case: strings can't be quoted/escaped at the
                # top level
                elif isinstance(val, gptypes.GPString):
                    query_dict[key] = val.value
                # Just use the wkid of SpatialReferences
                elif isinstance(val, geometry.SpatialReference): 
                    query_dict[key] = val.wkid
                # If it's a list, make it a comma-separated string
                elif isinstance(val, (list, tuple, set)):
                    val = ",".join([str(v.id) 
                                    if isinstance(v, Layer)
                                    else str(v) for v in val])
                # If it's a dictionary, dump as JSON
                elif isinstance(val, dict):
                    val = json.dumps(val)
                # Ignore null values, and coerce string values (hopefully
                # everything sent in to a query has a sane __str__)
                elif val is not None:
                    query_dict[key] = str(val)
        if self.__token__ is not None:
            query_dict['token'] = self.__token__
        query_dict[REQUEST_REFERER_MAGIC_NAME] = self._referer or self.url
        # Replace URL query component with newly altered component
        urllist[3] = compat.urlencode(query_dict)
        newurl = urllist
        # Instantiate new RestURL or subclass
        rt = returntype(newurl, file_data)
        # Remind the resource where it came from
        try:
            rt.parent = self
        except:
            rt._parent = self
        return rt
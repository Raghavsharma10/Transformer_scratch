def get_list_endpoint(self, rel=u"instances"):
        """
        get the configured list entpoint for the schema.type
        :param rel: lookup rel: value inside the links section
        :returns the value
        :raises APIException
        """
        schema_loaded = not self.schema is None
        links_present = "links" in self.schema.keys()
        if (schema_loaded and links_present):
             for row in self.schema['links']:
                  if row['rel'] == rel:
                      #print "row %s" % row
                      return row
        raise APIException("ENDPOINT_NOTFOUND","invalid endpoint")
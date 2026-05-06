def resolve_link(self, snow_record, field_to_resolve, **kparams):
        """
        Get the info from the link and return a SnowRecord.
        """
        try:
          link = snow_record.links()[field_to_resolve]
        except KeyError as e:
          return SnowRecord.NotFound(self, snow_record._table_name, "Could not find field %s in record" % field_to_resolve, [snow_record, field_to_resolve, self])

        if kparams:
            link += ('&', '?')[urlparse(link).query == '']
            link += '&'.join("%s=%s" % (key,val) for (key,val) in kparams.items())
        linked_response = self.req("get", link) # rety here...

        rjson = linked_response.json()
        rtablename = SnowRecord.tablename_from_link(link)

        # could do this, but better to not mutate:
        # setattr(snow_record, field_to_resolve, linked)
        # so just return new record. could infer

        if "result" in rjson:
            linked = SnowRecord(self, rtablename, **rjson["result"])
        else:
            linked = SnowRecord.NotFound(self, rtablename, "Could not resolve link %s" % link, [rjson, rtablename, link, self])

        return linked
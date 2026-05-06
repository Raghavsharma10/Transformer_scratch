def resolve_links(self, snow_record, **kparams):
        """
        Get the infos from the links and return SnowRecords[].
        """
        records = []
        for attr, link in snow_record.links().items():
            records.append(self.resolve_link(snow_record, attr, **kparams))
        return records
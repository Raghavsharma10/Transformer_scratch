def getTable(self, layer, where="1 = 1", fields=[], jsobj=None):
        """
        Returns JSON for a Table type. You shouldn't use this directly -- it's
        an automatic falback from .get if there is no geometry
        """
        base_where = where
        features = []
        # We always want to run once, and then break out as soon as we stop
        # getting exceededTransferLimit.
        while True:
            features += [feat.get('attributes') for feat in jsobj.get('features')]
            # There isn't an exceededTransferLimit?
            if len(jsobj.get('features')) < 1000:
                break
            # If we've hit the transfer limit we offset by the last OBJECTID
            # returned and keep moving along.
            where = "%s > %s" % (self.object_id_field, features[-1].get(self.object_id_field))
            if base_where != "1 = 1" :
                # If we have another WHERE filter we needed to tack that back on.
                where += " AND %s" % base_where
            jsobj = self.get_json(layer, where, fields)
        return features
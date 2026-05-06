def __rowResultToQuote(self, row):
        ''' convert rowResult from Hbase to Quote'''
        keyValues = row.columns
        for field in QUOTE_FIELDS:
            key = "%s:%s" % (HBaseDAM.QUOTE, field)
            if 'time' != field and keyValues[key].value:
                keyValues[key].value = float(keyValues[key].value)

        return Quote(*[keyValues["%s:%s" % (HBaseDAM.QUOTE, field)].value for field in QUOTE_FIELDS])
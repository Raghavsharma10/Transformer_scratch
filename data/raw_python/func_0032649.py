def __rowResultToTick(self, row):
        ''' convert rowResult from Hbase to Tick'''
        keyValues = row.columns
        for field in TICK_FIELDS:
            key = "%s:%s" % (HBaseDAM.TICK, field)
            if 'time' != field and keyValues[key].value:
                keyValues[key].value = float(keyValues[key].value)

        return Tick(*[keyValues["%s:%s" % (HBaseDAM.TICK, field)].value for field in TICK_FIELDS])
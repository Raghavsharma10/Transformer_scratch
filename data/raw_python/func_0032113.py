def createDAM(dam_name, config):
        ''' create DAM '''
        if 'yahoo' == dam_name:
            from analyzerdam.yahooDAM import YahooDAM
            dam=YahooDAM()
        elif 'google' == dam_name:
            from analyzerdam.google import GoogleDAM
            dam=GoogleDAM()
        elif 'excel' == dam_name:
            from analyzerdam.excelDAM import ExcelDAM
            dam=ExcelDAM()
        elif 'hbase' == dam_name:
            from analyzerdam.hbaseDAM import HBaseDAM
            dam=HBaseDAM()
        elif 'sql' == dam_name:
            from analyzerdam.sqlDAM import SqlDAM
            dam=SqlDAM(config)
        elif 'cex' == dam_name:
            from analyzerdam.cex import CexDAM
            dam=CexDAM(config)
        else:
            raise UfException(Errors.INVALID_DAM_TYPE,
                              "DAM type is invalid %s" % dam_name)

        return dam
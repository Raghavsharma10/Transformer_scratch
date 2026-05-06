def default(self, obj):
        """This is slightly different than json.JSONEncoder.default(obj)
        in that it should returned the serialized representation of the
        passed object, not a serializable representation.
        """
        if isinstance(obj, (datetime.date, datetime.time, datetime.datetime)):
            return '"%s"' % obj.isoformat()
        elif isinstance(obj, unicode):
            return '"%s"' % unicodedata.normalize('NFD', obj).encode('utf-8')
        elif isinstance(obj, decimal.Decimal):
            return str(obj)
        return super(Encoder, self).default(obj)
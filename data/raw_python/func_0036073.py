def update(self, ns, docid, raw, **kw):
        """ Perform a single update operation.

            {'docid': ObjectId('4e95ae3616692111bb000001'),
             'ns': u'mydb.tweets',
             'raw': {u'h': -5295451122737468990L,
                     u'ns': u'mydb.tweets',
                     u'o': {u'$set': {u'content': u'Lorem ipsum'}},
                     u'o2': {u'_id': ObjectId('4e95ae3616692111bb000001')},
                     u'op': u'u',
                     u'ts': Timestamp(1318432339, 1)}}
        """
        self._dest_coll(ns).update(raw['o2'], raw['o'], safe=True)
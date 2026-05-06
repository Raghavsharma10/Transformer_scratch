def delete(self, ns, docid, raw, **kw):
        """ Perform a single delete operation.

            {'docid': ObjectId('4e959ea11669210edc002902'),
             'ns': u'mydb.tweets',
             'raw': {u'b': True,
                     u'h': -8347418295715732480L,
                     u'ns': u'mydb.tweets',
                     u'o': {u'_id': ObjectId('4e959ea11669210edc002902')},
                     u'op': u'd',
                     u'ts': Timestamp(1318432261, 10499)}}
        """
        self._dest_coll(ns).remove(raw['o'], safe=True)
def flush(self, objects=None, batch_size=None, **kwargs):
        ''' flush objects stored in self.container or those passed in'''
        batch_size = batch_size or self.config.get('batch_size')
        # if we're flushing these from self.store, we'll want to
        # pop them later.
        if objects:
            from_store = False
        else:
            from_store = True
            objects = self.itervalues()
        # sort by _oid for grouping by _oid below
        objects = sorted(objects, key=lambda x: x['_oid'])
        batch, _ids = [], []
        # batch in groups with _oid, since upsert's delete
        # all _oid rows when autosnap=False!
        for key, group in groupby(objects, lambda x: x['_oid']):
            _grouped = list(group)
            if len(batch) + len(_grouped) > batch_size:
                logger.debug("Upserting %s objects" % len(batch))
                _ = self.upsert(objects=batch, **kwargs)
                logger.debug("... done upserting %s objects" % len(batch))
                _ids.extend(_)
                # start a new batch
                batch = _grouped
            else:
                # extend existing batch, since still will be < batch_size
                batch.extend(_grouped)
        else:
            if batch:
                # get the last batch too
                logger.debug("Upserting last batch of %s objects" % len(batch))
                _ = self.upsert(objects=batch, **kwargs)
                _ids.extend(_)
            logger.debug("... Finished upserting all objects!")

        if from_store:
            for _id in _ids:
                # try to pop the _id's flushed from store; warn / ignore
                # the KeyError if they're not there
                try:
                    self.store.pop(_id)
                except KeyError:
                    logger.warn(
                        "failed to pop {} from self.store!".format(_id))
        return sorted(_ids)
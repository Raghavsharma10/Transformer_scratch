def get_objects(self, flush=False, autosnap=True, **kwargs):
        '''
        Main API method for sub-classed cubes to override for the
        generation of the objects which are to (potentially) be added
        to the cube (assuming no duplicates)
        '''
        logger.debug('Running get_objects(flush=%s, autosnap=%s, %s)' % (
                     flush, autosnap, kwargs))
        if flush:
            s = time()
            result = self.flush(autosnap=autosnap, **kwargs)
            diff = time() - s
            logger.debug("Flush complete (%ss)" % int(diff))
            return result
        else:
            return self
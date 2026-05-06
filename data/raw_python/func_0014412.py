def add(self, item=None, finalize=False, callback=None):
        '''
        Takes a string, dictionary or list of items for adding to queue. To help troubleshoot it will output the updated buffer size, however when the content gets written it will output the file path of the new file. Generally this can be safely discarded.

        :param <dict,list> item: Item to add to the queue. If dict will be converted directly to a list and then to json. List must be a list of dictionaries. If a string is submitted, it will be written out as-is immediately and not buffered.
        :param bool finalize: If items are buffered internally, it will flush them to disk and return the file name.
        :param callback: A callback function that will be called when the item gets written to disk. It will be passed one position argument, the file path of the file written. Note that errors from the callback method will not be re-raised here.
        '''
        if item:
            if type(item) is list:
                check = list(set([type(d) for d in item]))
                if len(check) > 1 or dict not in check:
                    raise ValueError("More than one data type detected in item (list). Make sure they are all dicts of data going to Solr")
            elif type(item) is dict:
                item = [item]
            elif type(item) is str:
                return self._write_file(item)
            else:
                raise ValueError("Not the right data submitted. Make sure you are sending a dict or list of dicts")
        with self._rlock:
            res = self._preprocess(item, finalize, callback)
        return res
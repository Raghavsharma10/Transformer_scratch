def check_cache(self, type, data, obj=None):
        '''Returns the updated cached version of the given dict'''
        try:
            id = data['id']
        except:
            # Not an identifiable item
            #print 'don\'t know this item %r:%r' % (type, data)
            return data

        # If obj was passed in, its type takes precedence
        try:
            type = obj._get_type()
        except:
            pass

        # Find the item in the cache, update and return if it's there
        try:
            hit = self.item_cache[type][id]
        except KeyError:
            pass
        else:
            hit._update_data(data)
            #print 'cache hit for %s at %s' % (type, id)
            return hit

        # Not there? Let's make us a new item
        # If we weren't given the object ref, find the name in the global scope
        if not obj:
            # Default to Redmine_Item if it's not found
            obj = self.item_class.get(type, Redmine_Item)

        new_item = obj(redmine=self, data=data, type=type)

        # Store it
        self.item_cache.setdefault(type, {})[id] = new_item
        #print 'set new %s at %s' % (type, id)

        return new_item
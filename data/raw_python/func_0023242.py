def refresh_cache(self, cat_id):
        '''
        Repopulate cache
        '''        
        self.cache[cat_id] = most_recent_25_posts_by_category(cat_id)
        self.last_refresh[cat_id] = datetime.now()
        print ('Cache refresh at...', str(self.last_refresh[cat_id]))
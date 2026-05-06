def get_all_as_list(self, dir='_todo_dir'):
        '''
        Returns a list of the the full path to all items currently in the todo directory. The items will be listed in ascending order based on filesystem time.
        This will re-scan the directory on each execution.

        Do not use this to process items, this method should only be used for troubleshooting or something axillary. To process items use get_todo_items() iterator.
        '''
        dir = getattr(self,dir)
        list = [x for x in os.listdir(dir) if x.endswith('.json') or x.endswith('.json.gz')]
        full = [os.path.join(dir,x) for x in list]
        full.sort(key=lambda x: os.path.getmtime(x))
        return full
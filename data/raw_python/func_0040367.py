def thread_debug(self, *args, **kwargs):
        """
        Wrap debug to include thread information
        """
        if 'module' not in kwargs:
            kwargs['module'] = "Monitor"
        if kwargs['module'] != 'Monitor' and self.do_DEBUG(module='Monitor'):
            self.debug[kwargs['module']] = True
        if not self.do_DEBUG(module=kwargs['module']):
            return
        thread_id = threading.current_thread().name
        key = "[" + thread_id + "] " + kwargs['module']
        if not self.debug.get(key):
            self.debug[key] = True
        kwargs['module'] = key
        self.DEBUG(*args, **kwargs)
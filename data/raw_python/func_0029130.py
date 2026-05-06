def source_group_receiver(self, sender, source, signal, **kwargs):
        """
        Relay source group signals to the appropriate spec strategy.

        """
        from imagekit.cachefiles import ImageCacheFile
        source_group = sender

        instance = kwargs['instance']

        # Ignore signals from unregistered groups.
        if source_group not in self._source_groups:
            return

        #HOOK -- update source to point to image file.
        for id in self._source_groups[source_group]:

            spec_to_update = generator_registry.get(id, source=source, instance=instance, field=hack_spec_field_hash[id])            
                        
        specs = [generator_registry.get(id, source=source, instance=instance, field=hack_spec_field_hash[id]) for id in
                self._source_groups[source_group]]
        callback_name = self._signals[signal]
        # print 'callback_name? %s'%(callback_name)

        for spec in specs:
            file = ImageCacheFile(spec)
            # print 'SEPC %s file %s'%(spec, file)
            call_strategy_method(file, callback_name)
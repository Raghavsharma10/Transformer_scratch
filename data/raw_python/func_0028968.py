def source_group_receiver(self, sender, source, signal, **kwargs):
        """
        Relay source group signals to the appropriate spec strategy.

        """

        from imagekit.cachefiles import ImageCacheFile
        source_group = sender

        # Ignore signals from unregistered groups.
        if source_group not in self._source_groups:
            return


        #OVERRIDE HERE -- pass specs into generator object
        specs = [generator_registry.get(id, source=source, specs=spec_data_field_hash[id]) for id in
                self._source_groups[source_group]]
        callback_name = self._signals[signal]
        #END OVERRIDE

        for spec in specs:
            file = ImageCacheFile(spec)
            call_strategy_method(file, callback_name)
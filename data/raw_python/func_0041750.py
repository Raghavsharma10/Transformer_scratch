def pipeline(self, transaction=True, shard_hint=None):
        ''' Return a pipeline that support StoneRedis custom methods '''
        args_dict = {
            'connection_pool': self.connection_pool,
            'response_callbacks': self.response_callbacks,
            'transaction': transaction,
            'shard_hint': shard_hint,
            'logger': self.logger,
        }

        return StonePipeline(**args_dict)
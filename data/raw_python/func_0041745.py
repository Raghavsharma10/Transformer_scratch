def multi_rpush_limit(self, queue, values, limit=100000):
        ''' Pushes multiple elements to a list in an atomic way until it reaches certain size
            Once limit is reached, the function will lpop the oldest elements
            This operation runs in LUA, so is always atomic
        '''

        lua = '''
        local queue = KEYS[1]
        local max_size = tonumber(KEYS[2])
        local table_len = tonumber(table.getn(ARGV))
        local redis_queue_len = tonumber(redis.call('LLEN', queue))
        local total_size = redis_queue_len + table_len
        local from = 0

        if total_size >= max_size then
            -- Delete the same amount of data we are inserting. Even better, limit the queue to the specified size
            redis.call('PUBLISH', 'DEBUG', 'trim')
            if redis_queue_len - max_size + table_len > 0 then
                from = redis_queue_len - max_size + table_len
            else
                from = 0
            end
            redis.call('LTRIM', queue, from, redis_queue_len)
        end
        for _,key in ipairs(ARGV) do
            redis.call('RPUSH', queue, key)
        end
        return 1

        '''

        # Check that what we receive is iterable
        if hasattr(values, '__iter__'):
            if len(values) > limit:
                raise ValueError('The iterable size is bigger than the allowed limit ({1}): {0}'.format(len(values), limit))
            try:
                self.multi_rpush_limit_script([queue, limit], values)
            except AttributeError:
                if self.logger:
                    self.logger.info('Script not registered... registering')
                # If the script is not registered, register it
                self.multi_rpush_limit_script = self.register_script(lua)
                self.multi_rpush_limit_script([queue, limit], values)
        else:
            raise ValueError('Expected an iterable')
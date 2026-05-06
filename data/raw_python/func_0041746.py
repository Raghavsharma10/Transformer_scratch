def rpush_limit(self, queue, value, limit=100000):
        ''' Pushes an element to a list in an atomic way until it reaches certain size
            Once limit is reached, the function will lpop the oldest elements
            This operation runs in LUA, so is always atomic
        '''

        lua = '''
        local queue = KEYS[1]
        local max_size = tonumber(KEYS[2])
        local table_len = 1
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
        redis.call('RPUSH', queue, ARGV[1])
        return 1

        '''

        try:
            self.rpush_limit_script([queue, limit], [value])
        except AttributeError:
            if self.logger:
                self.logger.info('Script not registered... registering')
            # If the script is not registered, register it
            self.rpush_limit_script = self.register_script(lua)
            self.rpush_limit_script([queue, limit], [value])
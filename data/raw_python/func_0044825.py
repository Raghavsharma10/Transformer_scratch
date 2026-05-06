def _calc(count, 
              last_count, 
              start_time, 
              max_count, 
              speed_calc_cycles, 
              q, 
              last_speed,
              lock):
        """do the pre calculations in order to get TET, speed, TTG
        
        :param count:               count 
        :param last_count:          count at the last call, allows to treat the case of no progress
            between sequential calls
        :param start_time:          the time when start was triggered
        :param max_count:           the maximal value count 
        :type max_count:
        :param speed_calc_cycles:
        :type speed_calc_cycles:
        :param q:
        :type q:
        :param last_speed:
        :type last_speed:
        :param lock:
        :type lock:
        """
        count_value = count.value
        start_time_value = start_time.value
        current_time = time.time()
        
        if last_count.value != count_value:
            # some progress happened
        
            with lock:
                # save current state (count, time) to queue
                
                q.put((count_value, current_time))
    
                # get older state from queue (or initial state)
                # to to speed estimation                
                if q.qsize() > speed_calc_cycles:
                    old_count_value, old_time = q.get()
                else:
                    old_count_value, old_time = 0, start_time_value
            
            last_count.value = count_value
            #last_old_count.value = old_count_value
            #last_old_time.value = old_time
            
            speed = (count_value - old_count_value) / (current_time - old_time)
            last_speed.value = speed 
        else:
            # progress has not changed since last call
            # use also old (cached) data from the queue
            #old_count_value, old_time = last_old_count.value, last_old_time.value
            speed = last_speed.value  

        if (max_count is None):
            max_count_value = None
        else:
            max_count_value = max_count.value
            
        tet = (current_time - start_time_value)
        
        if (speed == 0) or (max_count_value is None) or (max_count_value == 0):
            ttg = None
        else:
            ttg = math.ceil((max_count_value - count_value) / speed)
            
        return count_value, max_count_value, speed, tet, ttg
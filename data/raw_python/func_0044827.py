def _show_stat(self):
        """
            convenient functions to call the static show_stat_wrapper_multi with
            the given class members
        """
        _show_stat_wrapper_multi_Progress(self.count,
                                          self.last_count, 
                                          self.start_time, 
                                          self.max_count, 
                                          self.speed_calc_cycles,
                                          self.width,
                                          self.q,
                                          self.last_speed,
                                          self.prepend,
                                          self.show_stat,
                                          self.len, 
                                          self.add_args,
                                          self.lock,
                                          self.info_line,
                                          no_move_up=True)
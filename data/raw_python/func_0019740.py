def retrieveVals(self):
        """Retrieve values for graphs."""
        if self.hasGraph('asterisk_calls') or self.hasGraph('asterisk_channels'):
            stats = self._ami.getChannelStats(self._chanList)
            if  self.hasGraph('asterisk_calls')  and stats:
                self.setGraphVal('asterisk_calls', 'active_calls', 
                                 stats.get('active_calls'))
                self.setGraphVal('asterisk_calls', 'calls_per_min', 
                                 stats.get('calls_processed'))
            if  self.hasGraph('asterisk_channels')  and stats:
                for field in self._chanList:
                    self.setGraphVal('asterisk_channels', 
                                     field, stats.get(field))
                if 'dahdi' in self._chanList:
                    self.setGraphVal('asterisk_channels', 
                                     'mix', stats.get('mix'))

        if self.hasGraph('asterisk_peers_sip'):
            stats = self._ami.getPeerStats('sip')
            if stats:
                for field in ('online', 'unmonitored', 'unreachable', 
                              'lagged', 'unknown'):
                    self.setGraphVal('asterisk_peers_sip', 
                                     field, stats.get(field))
        
        if self.hasGraph('asterisk_peers_iax2'):
            stats = self._ami.getPeerStats('iax2')
            if stats:
                for field in ('online', 'unmonitored', 'unreachable', 
                              'lagged', 'unknown'):
                    self.setGraphVal('asterisk_peers_iax2', 
                                     field, stats.get(field))
        
        if self.hasGraph('asterisk_voip_codecs'):
            sipstats = self._ami.getVoIPchanStats('sip', self._codecList) or {}
            iax2stats = self._ami.getVoIPchanStats('iax2', self._codecList) or {}
            if stats:
                for field in self._codecList:
                    self.setGraphVal('asterisk_voip_codecs', field,
                                     sipstats.get(field,0) 
                                     + iax2stats.get(field, 0))
                self.setGraphVal('asterisk_voip_codecs', 'other',
                                 sipstats.get('other', 0) 
                                 + iax2stats.get('other', 0))
        
        if self.hasGraph('asterisk_conferences'):
            stats = self._ami.getConferenceStats()
            if stats:
                self.setGraphVal('asterisk_conferences', 'rooms', 
                                 stats.get('active_conferences'))
                self.setGraphVal('asterisk_conferences', 'users', 
                                 stats.get('conference_users'))

        if self.hasGraph('asterisk_voicemail'):
            stats = self._ami.getVoicemailStats()
            if stats:
                self.setGraphVal('asterisk_voicemail', 'accounts', 
                                 stats.get('accounts'))
                self.setGraphVal('asterisk_voicemail', 'msg_avg', 
                                 stats.get('avg_messages'))
                self.setGraphVal('asterisk_voicemail', 'msg_max', 
                                 stats.get('max_messages'))
                self.setGraphVal('asterisk_voicemail', 'msg_total', 
                                 stats.get('total_messages'))

        if self.hasGraph('asterisk_trunks') and len(self._trunkList) > 0:
            stats = self._ami.getTrunkStats(self._trunkList)
            for trunk in self._trunkList:
                self.setGraphVal('asterisk_trunks', trunk[0], 
                                 stats.get(trunk[0]))
                
        if self._queues is not None:
            total_answer = 0
            total_abandon = 0
            for queue in self._queue_list:
                stats = self._queues[queue]
                if self.hasGraph('asterisk_queue_len'):
                    self.setGraphVal('asterisk_queue_len', queue,
                                     stats.get('queue_len'))
                if self.hasGraph('asterisk_queue_avg_hold'):
                    self.setGraphVal('asterisk_queue_avg_hold', queue,
                                     stats.get('avg_holdtime'))
                if self.hasGraph('asterisk_queue_avg_talk'):
                    self.setGraphVal('asterisk_queue_avg_talk', queue,
                                     stats.get('avg_talktime'))
                if self.hasGraph('asterisk_queue_calls'):
                    total_abandon += stats.get('calls_abandoned')
                    total_answer += stats.get('calls_completed')
                if self.hasGraph('asterisk_queue_abandon_pcent'):
                    prev_stats = self._queues_prev.get(queue)
                    if prev_stats is not None:
                        abandon = (stats.get('calls_abandoned', 0) -
                                   prev_stats.get('calls_abandoned', 0))
                        answer = (stats.get('calls_completed', 0) -
                                  prev_stats.get('calls_completed', 0))
                        total = abandon + answer    
                        if total > 0:
                            val = 100.0 * float(abandon) / float(total)
                        else:
                            val = 0
                        self.setGraphVal('asterisk_queue_abandon_pcent', 
                                     queue, val)
            if self.hasGraph('asterisk_queue_calls'):
                    self.setGraphVal('asterisk_queue_calls', 'abandon', 
                                     total_abandon)
                    self.setGraphVal('asterisk_queue_calls', 'answer', 
                                     total_answer)
                    
            if self.hasGraph('asterisk_fax_attempts'):
                fax_stats = self._ami.getFaxStatsCounters()
                stats = fax_stats.get('general')
                if stats is not None:
                    self.setGraphVal('asterisk_fax_attempts', 'send', 
                                     stats.get('transmit attempts'))
                    self.setGraphVal('asterisk_fax_attempts', 'recv', 
                                     stats.get('receive attempts'))
                    self.setGraphVal('asterisk_fax_attempts', 'fail', 
                                     stats.get('failed faxes'))
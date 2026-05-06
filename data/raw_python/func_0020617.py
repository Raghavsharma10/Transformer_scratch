def build_play(self, pbp_row):
        """
        Parses table row from RTSS. These are the rows tagged with ``<tr class='evenColor' ... >``. Result set
        contains :py:class:`nhlscrapi.games.playbyplay.Strength` and :py:class:`nhlscrapi.games.events.EventType`
        objects. Returned play data is in the form
        
        .. code:: python
        
            {
                'play_num': num_of_play
                'period': curr_period
                'strength': strength_enum
                'time': { 'min': min, 'sec': sec }
                'vis_on_ice': { 'player_num': player }
                'home_on_ice': { 'player_num': player }
                'event': event_object
            }
        
        :param pbp_row: table row from RTSS
        :returns: play data
        :rtype: dict
        """
        d = pbp_row.findall('./td')
        c = PlayParser.ColMap(self.season)
        
        p = { }
        to_dig = lambda t: int(t) if t.isdigit() else 0
        p['play_num'] = to_int(d[c["play_num"]].text, 0)
        p['period'] = to_int(d[c["per"]].text, 0)
        
        p['strength'] = self.__strength(d[c["str"]].text)
        
        time = d[c["time"]].text.split(":")
        p['time'] = { "min": int(time[0]), "sec": int(time[1]) }
        
        skater_tab = d[c["vis"]].xpath("./table")
        p['vis_on_ice'] = self.__skaters(skater_tab[0][0]) if len(skater_tab) else { }
            
        skater_tab = d[c["home"]].xpath("./table")
        p['home_on_ice'] = self.__skaters(skater_tab[0][0]) if len(skater_tab) else { }
            
        p['event'] = event_type_mapper(
            d[c["event"]].text,
            period=p['period'],
            skater_ct=len(p['vis_on_ice']) + len(p['home_on_ice']),
            game_type=self.game_type
        )
        p['event'].desc = " ".join([t.encode('ascii', 'replace').decode('utf-8') for t in d[c["desc"]].xpath("text()")])

        parse_event_desc(p['event'], season=self.season)
        
        return p
def __skaters(self, tab):
        """
        Constructs dictionary of players on the ice in the provided table at time of play.
        :param tab: RTSS table of the skaters and goalie on at the time of the play
        :rtype: dictionary, key = player number, value = [position, name]
        """
        
        res = { }
        for td in tab.iterchildren():
            if len(td):
                pl_data = td.xpath("./table/tr")
                pl = pl_data[0].xpath("./td/font")
                
                if pl[0].text.isdigit():
                    res[int(pl[0].text)] = [s.strip() for s in pl[0].get("title").split("-")][::-1]
                
                s = pl[0].get("title").split("-")
                pos = pl_data[1].getchildren()[0].text
                
        return res
def checkPos(self):
        """check all positions"""
        soup = BeautifulSoup(self.css1(path['movs-table']).html, 'html.parser')
        poss = []
        for label in soup.find_all("tr"):
            pos_id = label['id']
            # init an empty list
            # check if it already exist
            pos_list = [x for x in self.positions if x.id == pos_id]
            if pos_list:
                # and update it
                pos = pos_list[0]
                pos.update(label)
            else:
                pos = self.new_pos(label)
            pos.get_gain()
            poss.append(pos)
        # remove old positions
        self.positions.clear()
        self.positions.extend(poss)
        logger.debug("%d positions update" % len(poss))
        return self.positions
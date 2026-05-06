def parse_shifts(self):
        """
        Parse shifts from TOI report
        
        :returns: self if successfule else None
        """
        
        lx_doc = self.html_doc()
        pl_heads = lx_doc.xpath('//td[contains(@class, "playerHeading")]')
        for pl in pl_heads:
            sh_sum = { }
              
            pl_text = pl.xpath('text()')[0]
            num_name = pl_text.replace(',','').split(' ')
            sh_sum['player_num'] = int(num_name[0]) if num_name[0].isdigit() else -1
            sh_sum['player_name'] = { 'first': num_name[2], 'last': num_name[1] }
              
            first_shift = pl.xpath('../following-sibling::tr')[1]
            sh_sum['shifts'], last_shift = self.__player_shifts(first_shift)
              
            while ('Per' not in last_shift.xpath('.//text()')):
                last_shift = last_shift.xpath('following-sibling::tr')[0]
                
            per_summ = last_shift.xpath('.//tr')[0]
            sh_sum['by_period'], last_sum = self.__get_by_per_summ(per_summ)
            
            
            self.by_player[sh_sum['player_num']] = sh_sum
        
        return self if self.by_player else None
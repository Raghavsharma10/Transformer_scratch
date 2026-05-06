def inventory(self):
        """ Get module inventory. """

        self.m_info = self.get_attributes()
        if 'NOTCFP' in self.m_info['m_cfptype']:
            a = self.get_attribute('m_portcount')
            m_portcount = int(a)
        else:
            m_portcount = int(self.get_attribute('m_cfpconfig').split()[0])
        for p_index in range(m_portcount):
            XenaPort(parent=self, index='{}/{}'.format(self.index, p_index)).inventory()
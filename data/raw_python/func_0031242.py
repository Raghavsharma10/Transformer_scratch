def clear_port_stats(self):
        """ Clear only port stats (leave stream and packet group stats).

        Do not use - still working with Ixia to resolve.
        """
        stat = IxeStat(self)
        stat.ix_set_default()
        stat.enableValidStats = True
        stat.ix_set()
        stat.write()
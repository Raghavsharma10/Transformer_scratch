def remove_header(self, name):
        """Remove a field from the header"""
        if name in self.info_dict:
            self.info_dict.pop(name)
            logger.info("Removed '{0}' from INFO".format(name))
        if name in self.filter_dict:
            self.filter_dict.pop(name)
            logger.info("Removed '{0}' from FILTER".format(name))
        if name in self.format_dict:
            self.format_dict.pop(name)
            logger.info("Removed '{0}' from FORMAT".format(name))
        if name in self.contig_dict:
            self.contig_dict.pop(name)
            logger.info("Removed '{0}' from CONTIG".format(name))
        if name in self.alt_dict:
            self.alt_dict.pop(name)
            logger.info("Removed '{0}' from ALT".format(name))
        if name in self.other_dict:
            self.other_dict.pop(name)
            logger.info("Removed '{0}' from OTHER".format(name))
        return
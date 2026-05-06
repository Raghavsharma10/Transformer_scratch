def update_dict(self, base: Dict[str, LinkItem], new: Dict[str, LinkItem]):
        """
        Use for updating save state dicts and get the new save state dict. Provides a debug option at info level.
        Updates the base dict. Basically executes `base.update(new)`.

        :param base: base dict **gets overridden!**
        :type base: Dict[str, ~unidown.plugin.link_item.LinkItem]
        :param new: data which updates the base
        :type new: Dict[str, ~unidown.plugin.link_item.LinkItem]
        """
        if logging.INFO >= logging.getLevelName(dynamic_data.LOG_LEVEL):  # TODO: logging here or outside
            for link, item in new.items():
                if link in base:
                    self.log.info('Actualize item: ' + link + ' | ' + str(base[link]) + ' -> ' + str(item))
        base.update(new)
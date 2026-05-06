def get_updated_data(self, old_data: Dict[str, LinkItem]) -> Dict[str, LinkItem]:
        """
        Get links who needs to be downloaded by comparing old and the new data.

        :param old_data: old data
        :type old_data: Dict[str, ~unidown.plugin.link_item.LinkItem]
        :return: data which is newer or dont exist in the old one
        :rtype: Dict[str, ~unidown.plugin.link_item.LinkItem]
        """
        if not self.download_data:
            return {}
        new_link_item_dict = {}
        for link, link_item in tqdm(self.download_data.items(), desc="Compare with save", unit="item", leave=True,
                                    mininterval=1, ncols=100, disable=dynamic_data.DISABLE_TQDM):
            # TODO: add methode to log lost items, which are in old but not in new
            # if link in new_link_item_dict:  # TODO: is ever false, since its the key of a dict: move to the right place
            # self.log.warning("Duplicate: " + link + " - " + new_link_item_dict[link] + " : " + link_item)

            # if the new_data link does not exists in old_data or new_data time is newer
            if (link not in old_data) or (link_item.time > old_data[link].time):
                new_link_item_dict[link] = link_item

        return new_link_item_dict
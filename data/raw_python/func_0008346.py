def check_download(self, link_item_dict: Dict[str, LinkItem], folder: Path, log: bool = True) -> Tuple[
        Dict[str, LinkItem], Dict[str, LinkItem]]:
        """
        Check if the download of the given dict was successful. No proving if the content of the file is correct too.

        :param link_item_dict: dict which to check
        :type link_item_dict: Dict[str, ~unidown.plugin.link_item.LinkItem]
        :param folder: folder where the downloads are saved
        :type folder: ~pathlib.Path
        :param log: if the lost items should be logged
        :type log: bool
        :return: succeeded and lost dicts
        :rtype: Tuple[Dict[str, ~unidown.plugin.link_item.LinkItem], Dict[str, ~unidown.plugin.link_item.LinkItem]]
        """
        succeed = {link: item for link, item in link_item_dict.items() if folder.joinpath(item.name).is_file()}
        lost = {link: item for link, item in link_item_dict.items() if link not in succeed}

        if lost and log:
            for link, item in lost.items():
                self.log.error(f"Not downloaded: {self.info.host+link} - {item.name}")

        return succeed, lost
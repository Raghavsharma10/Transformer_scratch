def download(self, link_item_dict: Dict[str, LinkItem], folder: Path, desc: str, unit: str, delay: float = 0) -> \
            List[str]:
        """
        .. warning::

            The parameters may change in future versions. (e.g. change order and accept another host)

        Download the given LinkItem dict from the plugins host, to the given path. Proceeded with multiple connections
        :attr:`~unidown.plugin.a_plugin.APlugin._simul_downloads`. After
        :func:`~unidown.plugin.a_plugin.APlugin.check_download` is recommend.

        This function don't use an internal `link_item_dict`, `delay` or `folder` directly set in options or instance
        vars, because it can be used aside of the normal download routine inside the plugin itself for own things.
        As of this it still needs access to the logger, so a staticmethod is not possible.

        :param link_item_dict: data which gets downloaded
        :type link_item_dict: Dict[str, ~unidown.plugin.link_item.LinkItem]
        :param folder: target download folder
        :type folder: ~pathlib.Path
        :param desc: description of the progressbar
        :type desc: str
        :param unit: unit of the download, shown in the progressbar
        :type unit: str
        :param delay: delay between the downloads in seconds
        :type delay: float
        :return: list of urls of downloads without errors
        :rtype: List[str]
        """
        if 'delay' in self._options:
            delay = self._options['delay']
        # TODO: add other optional host?
        if not link_item_dict:
            return []

        job_list = []
        with ThreadPoolExecutor(max_workers=self._simul_downloads) as executor:
            for link, item in link_item_dict.items():
                job = executor.submit(self.download_as_file, link, folder, item.name, delay)
                job_list.append(job)

            pbar = tqdm(as_completed(job_list), total=len(job_list), desc=desc, unit=unit, leave=True, mininterval=1,
                        ncols=100, disable=dynamic_data.DISABLE_TQDM)
            for _ in pbar:
                pass

        download_without_errors = []
        for job in job_list:
            try:
                download_without_errors.append(job.result())
            except HTTPError as ex:
                self.log.warning("Failed to download: " + str(ex))
                # Todo: connection lost handling (check if the connection to the server itself is lost)

        return download_without_errors
def get_rsc_list_2(self, rsc_clz_list=None):
        """get the list of resource list to collect based on clz list

        :param rsc_clz_list: the list of classes to collect
        :return: filtered list of resource list,
                 like [VNXLunList(), VNXDiskList()]
        """
        rsc_list_2 = self._default_rsc_list_with_perf_stats()
        if rsc_clz_list is None:
            rsc_clz_list = ResourceList.get_rsc_clz_list(rsc_list_2)

        return [rsc_list
                for rsc_list in rsc_list_2
                if rsc_list.get_resource_class() in rsc_clz_list]
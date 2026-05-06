def _get_workflow_menus(self):
        """
        Creates menu entries for custom workflows.

        Returns:
            Dict of list of dicts (``{'':[{}],}``). Menu entries.
        """
        results = defaultdict(list)
        from zengine.lib.cache import WFSpecNames
        for name, title, category in WFSpecNames().get_or_set():
            if self.current.has_permission(name) and category != 'hidden':
                wf_dict = {
                    "text": title,
                    "wf": name,
                    "kategori": category,
                    "param": "id"
                }
                results['other'].append(wf_dict)
                self._add_to_quick_menu(name, wf_dict)
        return results
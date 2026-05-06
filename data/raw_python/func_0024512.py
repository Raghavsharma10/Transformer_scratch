def get_dashboard_full(self, db_name):
        """Get DB and all objs needed to duplicate it"""
        objects = {}
        dashboards = self.get_objects("type", "dashboard")
        vizs = self.get_objects("type", "visualization")
        searches = self.get_objects("type", "search")
        if db_name not in dashboards:
            return None
        self.pr_inf("Found dashboard: " + db_name)
        objects[db_name] = dashboards[db_name]
        panels = json.loads(dashboards[db_name]['_source']['panelsJSON'])
        for panel in panels:
            if 'id' not in panel:
                continue
            pid = panel['id']
            if pid in searches:
                self.pr_inf("Found search:    " + pid)
                objects[pid] = searches[pid]
            elif pid in vizs:
                self.pr_inf("Found vis:       " + pid)
                objects[pid] = vizs[pid]
                emb = vizs[pid].get('_source', {}).get('savedSearchId', None)
                if emb is not None and emb not in objects:
                    if emb not in searches:
                        self.pr_err('Missing search %s' % emb)
                        return objects
                    objects[emb] = searches[emb]
        return objects
def run_svc_action(self, name, replace=None, svc=None):
        """
        backwards compatible to reflex service object. This looks for hooks on
        current object as well as in the actions sub-object.
        """
        actions = svc.get('actions')
        if actions and actions.get(name):
            return self.run(name, actions=actions, replace=replace)
        if svc.get(name + "-hook"):
            return self.run(name, actions={
                name: {
                    "type": "hook",
                    "url": svc.get(name + "-hook")
                }
            }, replace=replace)
        self.die("Unable to find action {name} on service {svc}",
                 name=name, svc=svc.get('name', ''))
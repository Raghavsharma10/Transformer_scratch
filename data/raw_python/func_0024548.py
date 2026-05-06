def create_wf_instances(self, roles=None):
        """
        Creates wf instances.
        Args:
            roles (list): role list

        Returns:
            (list): wf instances
        """

        # if roles specified then create an instance for each role
        # else create only one instance

        if roles:
            wf_instances = [
                WFInstance(
                    wf=self.wf,
                    current_actor=role,
                    task=self,
                    name=self.wf.name
                ) for role in roles
                ]
        else:
            wf_instances = [
                WFInstance(
                    wf=self.wf,
                    task=self,
                    name=self.wf.name
                )
            ]

        # if task type is not related with objects save instances immediately.
        if self.task_type in ["C", "D"]:
            return [wfi.save() for wfi in wf_instances]

        # if task type is related with its objects, save populate instances per object
        else:
            wf_obj_instances = []
            for wfi in wf_instances:
                role = wfi.current_actor if self.task_type == "A" else None
                keys = self.get_object_keys(role)
                wf_obj_instances.extend(
                    [WFInstance(
                        wf=self.wf,
                        current_actor=role,
                        task=self,
                        name=self.wf.name,
                        wf_object=key,
                        wf_object_type=self.object_type
                    ).save() for key in keys]
                )

            return wf_obj_instances
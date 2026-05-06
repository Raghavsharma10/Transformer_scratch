def get_instance_subnet_name(self, instance):
        """
        Return a human readable name for given instance's subnet, or None.

        Uses stored config mapping of subnet IDs to names.
        """
        # TODO: we have to do this here since we are monkeypatching Instance.
        # If we switch to custom Instance (sub)class then we could do it in the
        # object, provided it has access to the configuration data.
        if instance.subnet_id:
            # Account for omitted 'subnet-'
            subnet = self.config['subnets'][instance.subnet_id[7:]]
        else:
            subnet = BLANK
        return subnet
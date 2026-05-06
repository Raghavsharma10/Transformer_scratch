def update_region_to_controller(self, region):
        """
        Update the controller string with dynamic region info.
        Controller string should end up as `<name[-env]>.<region>.cloudgenix.com`

        **Parameters:**

          - **region:** region string.

        **Returns:** No return value, mutates the controller in the class namespace
        """
        # default region position in a list
        region_position = 1

        # Check for a global "ignore region" flag
        if self.ignore_region:
            # bypass
            api_logger.debug("IGNORE_REGION set, not updating controller region.")
            return

        api_logger.debug("Updating Controller Region")
        api_logger.debug("CONTROLLER = %s", self.controller)
        api_logger.debug("CONTROLLER_ORIG = %s", self.controller_orig)
        api_logger.debug("CONTROLLER_REGION = %s", self.controller_region)

        # Check if this is an initial region use or an update region use
        if self.controller_orig:
            controller_base = self.controller_orig
        else:
            controller_base = self.controller
            self.controller_orig = self.controller

        # splice controller string
        controller_full_part_list = controller_base.split('.')

        for idx, part in enumerate(controller_full_part_list):
            # is the region already in the controller string?
            if region == part:
                # yes, controller already has apropriate region
                api_logger.debug("REGION %s ALREADY IN CONTROLLER AT INDEX = %s", region, idx)
                # update region if it is not already set.
                if self.controller_region != region:
                    self.controller_region = region
                    api_logger.debug("UPDATED_CONTROLLER_REGION = %s", self.controller_region)
                return

        controller_part_count = len(controller_full_part_list)

        # handle short domain case
        if controller_part_count > 1:
            # insert region
            controller_full_part_list[region_position] = region
            self.controller = ".".join(controller_full_part_list)
        else:
            # short domain, just add region
            self.controller = ".".join(controller_full_part_list) + '.' + region

        # update SDK vars with region info
        self.controller_orig = controller_base
        self.controller_region = region

        api_logger.debug("UPDATED_CONTROLLER = %s", self.controller)
        api_logger.debug("UPDATED_CONTROLLER_ORIG = %s", self.controller_orig)
        api_logger.debug("UPDATED_CONTROLLER_REGION = %s", self.controller_region)
        return
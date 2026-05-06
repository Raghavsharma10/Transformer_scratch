def set_scenario_role_names(self):
        """Populates the list of scenario role names in this deployment and
        populates the scenario_master with the master role

        Gets a list of deployment properties containing "isMaster" because
        there is exactly one per scenario host, containing the role name

        :return:
        """
        log = logging.getLogger(self.cls_logger + '.set_scenario_role_names')
        is_master_props = self.get_matching_property_names('isMaster')
        for is_master_prop in is_master_props:
            role_name = is_master_prop.split('.')[-1]
            log.info('Adding scenario host: {n}'.format(n=role_name))
            self.scenario_role_names.append(role_name)

            # Determine if this is the scenario master
            is_master = self.get_value(is_master_prop).lower().strip()
            if is_master == 'true':
                log.info('Found master scenario host: {r}'.format(r=role_name))
                self.scenario_master = role_name
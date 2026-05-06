def get_rds_instances_by_region(self, region):
        ''' Makes an AWS API call to the list of RDS instances in a particular
        region '''

        try:
            conn = self.connect_to_aws(rds, region)
            if conn:
                instances = conn.get_all_dbinstances()
                for instance in instances:
                    self.add_rds_instance(instance, region)
        except boto.exception.BotoServerError as e:
            error = e.reason

            if e.error_code == 'AuthFailure':
                error = self.get_auth_error_message()
            if not e.reason == "Forbidden":
                error = "Looks like AWS RDS is down:\n%s" % e.message
            self.fail_with_error(error, 'getting RDS instances')
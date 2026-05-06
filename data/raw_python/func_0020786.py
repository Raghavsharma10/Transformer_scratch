def listPhysicsGroups(self, physics_group_name=''):
        """
        API to list all physics groups.

        :param physics_group_name: List that specific physics group (Optional)
        :type physics_group_name: basestring
        :returns: List of dictionaries containing the following key (physics_group_name)
        :rtype: list of dicts

        """
        if physics_group_name:
            physics_group_name = physics_group_name.replace('*', '%')
        try:
            return self.dbsPhysicsGroup.listPhysicsGroups(physics_group_name)
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.serverError)
        except Exception as ex:
            sError = "DBSReaderModel/listPhysicsGroups. %s\n. Exception trace: \n %s" \
                    % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error', dbsExceptionCode['dbsException-server-error'], self.logger.exception, sError)
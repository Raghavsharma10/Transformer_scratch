def orderrun_detail(dk_api, kitchen, pd):
        """
        returns a string.
        :param dk_api: -- api object
        :param kitchen: string
        :param pd: dict
        :rtype: DKReturnCode
        """
        if DKCloudCommandRunner.SUMMARY in pd:
            display_summary = True
        else:
            display_summary = False
        # always get summary information
        pd[DKCloudCommandRunner.SUMMARY] = True
        rc = dk_api.orderrun_detail(kitchen, pd)
        s = ''
        if not rc.ok() or not isinstance(rc.get_payload(), list):
            s = 'Issue with getting order run details\nmessage: %s' % rc.get_message()
            rc.set_message(s)
            return rc

        # we have a list of servings, find the right dict
        serving_list = rc.get_payload()
        serving = None
        if DKCloudCommandRunner.ORDER_RUN_ID in pd:
            order_run_id = pd[DKCloudCommandRunner.ORDER_RUN_ID]
            for serv in serving_list:
                if serv[DKCloudCommandRunner.ORDER_RUN_ID] == order_run_id:
                    serving = serv
                    break
        elif DKCloudCommandRunner.ORDER_ID in pd:
            order_id = pd[DKCloudCommandRunner.ORDER_ID]
            for serv in serving_list:
                if serv[DKCloudCommandRunner.ORDER_ID] == order_id:
                    serving = serv
                    break
        else:
            # find the newest serving
            dex = -1
            latest = None
            for i, serving in enumerate(serving_list):
                if DKCloudCommandRunner.ORDER_ID in serving and serving[DKCloudCommandRunner.ORDER_ID] > latest:
                    latest = serving[DKCloudCommandRunner.ORDER_ID]
                    dex = i
            if dex != -1:
                serving = serving_list[dex]

        if serving is None:
            rc.set(rc.DK_FAIL,
                   "No OrderRun information.  Try using 'dk order-list -k %s' to see what is available." % kitchen)
            return rc

        # serving now contains the dictionary of the serving to display
        # pull out the information and put it in the message string of the rc

        if serving and display_summary:
            s += '\nORDER RUN SUMMARY\n\n'
            summary = None
            if DKCloudCommandRunner.SUMMARY in serving:
                summary = serving[DKCloudCommandRunner.SUMMARY]
            pass
            s += 'Order ID:\t%s\n' % serving[DKCloudCommandRunner.ORDER_ID]
            orid_from_serving = serving[DKCloudCommandRunner.ORDER_RUN_ID]
            s += 'Order Run ID:\t%s\n' % orid_from_serving
            s += 'Status:\t\t%s\n' % serving['status']
            s += 'Kitchen:\t%s\n' % kitchen

            if summary and 'name' in summary:
                s += 'Recipe:\t\t%s\n' % summary['name']
            else:
                s += 'Recipe:\t\t%s\n' % 'Not available'

            # variation name is inside the order id, pull it out
            s += 'Variation:\t%s\n' % orid_from_serving.split('#')[3]

            if summary and 'start-time' in summary:
                start_time = summary['start-time']
                if isinstance(start_time, basestring):
                    s += 'Start time:\t%s\n' % summary['start-time'].split('.')[0]
                else:
                    s += 'Start time:\t%s\n' % 'Not available 1'
            else:
                s += 'Start time:\t%s\n' % 'Not available 2'

            run_time = None
            if summary and 'total-recipe-time' in summary:
                run_time = summary['total-recipe-time']
            if isinstance(run_time, basestring):  # Active recipes don't have a run-duration
                s += 'Run duration:\t%s (H:M:S)\n' % run_time.split('.')[0]
            else:
                s += 'Run duration:\t%s\n' % 'Not available'

        if serving and DKCloudCommandRunner.TESTRESULTS in serving and \
                isinstance(serving[DKCloudCommandRunner.TESTRESULTS], basestring):
            s += '\nTEST RESULTS'
            s += serving[DKCloudCommandRunner.TESTRESULTS]
        if serving and DKCloudCommandRunner.TIMINGRESULTS in serving and \
                isinstance(serving[DKCloudCommandRunner.TIMINGRESULTS], basestring):
            s += '\n\nTIMING RESULTS\n\n'
            s += serving[DKCloudCommandRunner.TIMINGRESULTS]
        if serving and DKCloudCommandRunner.LOGS in serving and \
                isinstance(serving[DKCloudCommandRunner.LOGS], basestring):
            s += '\n\nLOG\n\n'
            s += DKCloudCommandRunner._decompress(serving[DKCloudCommandRunner.LOGS])
        if 'status' in pd and serving and DKCloudCommandRunner.SUMMARY in serving and \
                isinstance(serving[DKCloudCommandRunner.SUMMARY], dict):
            s += '\nSTEP STATUS\n\n'
            summary = serving[DKCloudCommandRunner.SUMMARY]
            # loop through the sorted keys
            for key in sorted(summary):
                value = summary[key]
                if isinstance(value, dict):
                    # node/step info is stored as a dictionary, print the node name (key) and status
                    if 'status' in value:
                        status = value['status']
                    else:
                        status = 'unknown'
                    s += '%s\t%s\n' % (key, status)

        if serving and 'runstatus' in pd:
            s += serving['status']

        if serving and 'disp_order_id' in pd and DKCloudCommandRunner.ORDER_ID in serving:
            s += serving[DKCloudCommandRunner.ORDER_ID]

        if serving and 'disp_order_run_id' in pd and DKCloudCommandRunner.ORDER_RUN_ID in serving:
            s += serving[DKCloudCommandRunner.ORDER_RUN_ID]

        rc.set_message(s)
        return rc
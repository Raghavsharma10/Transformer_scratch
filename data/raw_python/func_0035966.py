def orderrun_detail(backend, kitchen, summary, nodestatus, runstatus, log, timing, test, all_things,
                    order_id, order_run_id, disp_order_id, disp_order_run_id):
    """
    Display information about an Order-Run
    """
    err_str, use_kitchen = Backend.get_kitchen_from_user(kitchen)
    if use_kitchen is None:
        raise click.ClickException(err_str)
    # if recipe is None:
    #     recipe = DKRecipeDisk.find_reciper_name()
    #     if recipe is None:
    #         raise click.ClickException('You must be in a recipe folder, or provide a recipe name.')
    pd = dict()
    if all_things:
        pd['summary'] = True
        pd['logs'] = True
        pd['timingresults'] = True
        pd['testresults'] = True
        # pd['state'] = True
        pd['status'] = True
    if summary:
        pd['summary'] = True
    if log:
        pd['logs'] = True
    if timing:
        pd['timingresults'] = True
    if test:
        pd['testresults'] = True
    if nodestatus:
        pd['status'] = True

    if runstatus:
        pd['runstatus'] = True
    if disp_order_id:
        pd['disp_order_id'] = True
    if disp_order_run_id:
        pd['disp_order_run_id'] = True

    # if the user does not specify anything to display, show the summary information
    if not runstatus and \
            not all_things and \
            not test and \
            not timing and \
            not log and \
            not nodestatus and \
            not summary and \
            not disp_order_id and \
            not disp_order_run_id:
        pd['summary'] = True

    if order_id is not None and order_run_id is not None:
        raise click.ClickException("Cannot specify both the Order Id and the OrderRun Id")
    if order_id is not None:
        pd[DKCloudCommandRunner.ORDER_ID] = order_id.strip()
    elif order_run_id is not None:
        pd[DKCloudCommandRunner.ORDER_RUN_ID] = order_run_id.strip()

    # don't print the green thing if it is just runstatus
    if not runstatus and not disp_order_id and not disp_order_run_id:
        click.secho('%s - Display Order-Run details from kitchen %s' % (get_datetime(), use_kitchen), fg='green')
    check_and_print(DKCloudCommandRunner.orderrun_detail(backend.dki, use_kitchen, pd))
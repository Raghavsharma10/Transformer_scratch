def delete_perf_task(task_name, auth, url):
    """
    Function takes a str of the target task_name to be deleted and retrieves task_id using
    the get_perf_task function. Once the task_id has been successfully retrieved it is
    populated into the task_id variable and an DELETE call is made against the HPE IMC REST
    interface to delete the target task.
    :param task_name: str of task name
    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: int of 204 if successful, str of "Perf Task doesn't exist" i

    :rtype: int

    """
    task_id = get_perf_task(task_name, auth, url)
    if isinstance(task_id, str):
        print("Perf task doesn't exist")
        return 403
    task_id = task_id['taskId']
    get_perf_task_url = "/imcrs/perf/task/delete/" + str(task_id)
    f_url = url + get_perf_task_url
    response = requests.delete(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 204:
            print("Perf Task successfully delete")
            return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + ' delete_perf_task: An Error has occured'
def get_perf_task(task_name, auth, url):
    """
        function takes the a str object containing the name of an existing performance tasks and
        issues a RESTFUL call to the IMC REST service. It will return a list

        :param task_name: str containing the name of the performance task

        :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

        :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

        :return: 204

        :rtype: dict

        >>> from pyhpeimc.auth import *

        >>> from pyhpeimc.plat.perf import *

        >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

        >>> selected_task = get_perf_task('Cisco_Temperature', auth.creds, auth.url)

        >>> assert type(selected_task) is dict

        >>> assert 'taskName' in selected_task
        """
    get_perf_task_url = "/imcrs/perf/task?name=" + task_name + "&orderBy=taskId&desc=false"
    f_url = url + get_perf_task_url
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            perf_task_info = (json.loads(response.text))
            if 'task' in perf_task_info:
                perf_task_info = (json.loads(response.text))['task']
            else:
                perf_task_info = "Task Doesn't Exist"
            return perf_task_info
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + ' get_perf_task: An Error has occured'
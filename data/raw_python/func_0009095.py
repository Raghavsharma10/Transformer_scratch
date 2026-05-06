def send_build_close(params,response_url):
    '''send build close sends a final response (post) to the server to bring down
    the instance. The following must be included in params:

    repo_url, logfile, repo_id, secret, log_file, token
    '''
    # Finally, package everything to send back to shub
    response = {"log": json.dumps(params['log_file']),
                "repo_url": params['repo_url'],
                "logfile": params['logfile'],
                "repo_id": params['repo_id'],
                "container_id": params['container_id']}

    body = '%s|%s|%s|%s|%s' %(params['container_id'],
                              params['commit'],
                              params['branch'],
                              params['token'],
                              params['tag']) 

    signature = generate_header_signature(secret=params['token'],
                                          payload=body,
                                          request_type="finish")

    headers = {'Authorization': signature }

    finish = requests.post(response_url,data=response, headers=headers)
    bot.debug("FINISH POST TO SINGULARITY HUB ---------------------")
    bot.debug(finish.status_code)
    bot.debug(finish.reason)
    return finish
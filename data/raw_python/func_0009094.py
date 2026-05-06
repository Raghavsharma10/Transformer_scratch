def send_build_data(build_dir, data, secret, 
                    response_url=None,clean_up=True):
    '''finish build sends the build and data (response) to a response url
    :param build_dir: the directory of the build
    :response_url: where to send the response. If None, won't send
    :param data: the data object to send as a post
    :param clean_up: If true (default) removes build directory
    '''
    # Send with Authentication header
    body = '%s|%s|%s|%s|%s' %(data['container_id'],
                              data['commit'],
                              data['branch'],
                              data['token'],
                              data['tag']) 

    signature = generate_header_signature(secret=secret,
                                          payload=body,
                                          request_type="push")

    headers = {'Authorization': signature }

    if response_url is not None:
        finish = requests.post(response_url,data=data, headers=headers)
        bot.debug("RECEIVE POST TO SINGULARITY HUB ---------------------")
        bot.debug(finish.status_code)
        bot.debug(finish.reason)
    else:
        bot.warning("response_url set to None, skipping sending of build.")

    if clean_up == True:
        shutil.rmtree(build_dir)

    # Delay a bit, to give buffer between bringing instance down
    time.sleep(20)
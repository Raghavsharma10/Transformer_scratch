def get_json(uri):
    """
    Handle headers and json for us :3
    """
    response = requests.get(API + uri, headers=HEADERS)

    limit = int(response.headers.get("x-ratelimit-remaining"))
    if limit == 0:
        sys.stdout.write("\n")
        message = "You have run out of GitHub request tokens. "

        if int(response.headers.get("x-ratelimit-limit")) == 60:
            message += "Set a GITHUB_TOKEN to increase your limit to 5000/hour. "

        wait_seconds = int(response.headers.get("x-ratelimit-reset")) - int(time.time())
        wait_minutes = math.ceil(wait_seconds / 60)
        message += "Try again in ~%d minutes. " % wait_minutes

        if "--wait-for-reset" in sys.argv:
            progress_message(message.replace("Try ", "Trying "))
            time.sleep(wait_seconds + 1)
            progress_message("Resuming")
            return get_json(uri)
        else:
            raise ValueError(message)

    progress()
    return response.json()
def main():
    """Handles external calling for this module

    Execute this python module and provide the args shown below to
    external call this module to send Slack messages with attachments!

    :return: None
    """
    log = logging.getLogger(mod_logger + '.main')
    parser = argparse.ArgumentParser(description='This Python module allows '
                                                 'sending Slack messages.')
    parser.add_argument('-u', '--url', help='Slack webhook URL', required=True)
    parser.add_argument('-t', '--text', help='Text of the message', required=True)
    parser.add_argument('-n', '--channel', help='Slack channel', required=True)
    parser.add_argument('-i', '--icon', help='URL for the Slack icon', required=False)
    parser.add_argument('-c', '--color', help='Color of the Slack post', required=False)
    parser.add_argument('-a', '--attachment', help='Text for the Slack Attachment', required=False)
    parser.add_argument('-p', '--pretext', help='Pretext for the Slack attachment', required=False)
    args = parser.parse_args()

    # Create the SlackMessage object
    try:
        slack_msg = SlackMessage(args.url, channel=args.channel, icon_url=args.icon, text=args.text)
    except ValueError as e:
        msg = 'Unable to create slack message\n{ex}'.format(ex=e)
        log.error(msg)
        print(msg)
        return

    # If provided, create the SlackAttachment object
    if args.attachment:
        try:
            slack_att = SlackAttachment(fallback=args.attachment, color=args.color,
                                        pretext=args.pretext, text=args.attachment)
        except ValueError:
            _, ex, trace = sys.exc_info()
            log.error('Unable to create slack attachment\n{e}'.format(e=str(ex)))
            return
        slack_msg.add_attachment(slack_att)

    # Send Slack message
    try:
        slack_msg.send()
    except(TypeError, ValueError, IOError):
        _, ex, trace = sys.exc_info()
        log.error('Unable to send Slack message\n{e}'.format(e=str(ex)))
        return
    log.debug('Your message has been Slacked successfully!')
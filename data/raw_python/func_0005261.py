def main():
    """Handles external calling for this module

    Execute this python module and provide the args shown below to
    external call this module to send email messages!

    :return: None
    """
    log = logging.getLogger(mod_logger + '.main')
    parser = argparse.ArgumentParser(description='This module allows sending email messages.')
    parser.add_argument('-f', '--file', help='Full path to a plain text file', required=False)
    parser.add_argument('-s', '--sender', help='Email address of the sender', required=False)
    parser.add_argument('-r', '--recipient', help='Email address of the recipient', required=False)
    args = parser.parse_args()

    am = AssetMailer()
    err = None
    if args.file:
        try:
            am.send_text_file(text_file=args.file, sender=args.sender, recipient=args.recipient)
        except AssetMailerError:
            _, ex, trace = sys.exc_info()
            err = '{n}: There was a problem sending email with file {f} from sender {s} to recipient {r}:\n{e}'.format(
                n=ex.__class__.__name__, f=args.file, s=args.sender, r=args.recipient, e=str(ex))
            log.error(err)
    else:
        try:
            am.send_cons3rt_agent_logs()
        except AssetMailerError:
            _, ex, trace = sys.exc_info()
            err = '{n}: There was a problem sending cons3rt agent log files:\n{e}'.format(
                n=ex.__class__.__name__, e=str(ex))
            log.error(err)
    if err is None:
        log.info('Successfully send email')
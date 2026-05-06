def create_parser():
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description='Sync a local and a remote folder through SFTP.'
    )

    parser.add_argument(
        "path",
        type=str,
        metavar="local-path",
        help="the path of the local folder",
    )

    parser.add_argument(
        "remote",
        type=str,
        metavar="user[:password]@hostname:remote-path",
        help="the ssh-url ([user[:password]@]hostname:remote-path) of the remote folder. "
             "The hostname can be specified as a ssh_config's hostname too. "
             "Every missing information will be gathered from there",
    )

    parser.add_argument(
        "-k",
        "--key",
        metavar="identity-path",
        action="append",
        help="private key identity path (defaults to ~/.ssh/id_rsa)"
    )

    parser.add_argument(
        "-l",
        "--logging",
        choices=['CRITICAL',
                 'ERROR',
                 'WARNING',
                 'INFO',
                 'DEBUG',
                 'NOTSET'],
        default='ERROR',
        help="set logging level"
    )

    parser.add_argument(
        "-p",
        "--port",
        default=22,
        type=int,
        help="SSH remote port (defaults to 22)"
    )

    parser.add_argument(
        "-f",
        "--fix-symlinks",
        action="store_true",
        help="fix symbolic links on remote side"
    )

    parser.add_argument(
        "-a",
        "--ssh-agent",
        action="store_true",
        help="enable ssh-agent support"
    )

    parser.add_argument(
        "-c",
        "--ssh-config",
        metavar="ssh_config path",
        default="~/.ssh/config",
        type=str,
        help="path to the ssh-configuration file (default to ~/.ssh/config)"
    )

    parser.add_argument(
        "-n",
        "--known-hosts",
        metavar="known_hosts path",
        default="~/.ssh/known_hosts",
        type=str,
        help="path to the openSSH known_hosts file"
    )

    parser.add_argument(
        "-d",
        "--disable-known-hosts",
        action="store_true",
        help="disable known_hosts fingerprint checking (security warning!)"
    )

    parser.add_argument(
        "-e",
        "--exclude-from",
        metavar="exclude-from-file-path",
        type=str,
        help="exclude files matching pattern in exclude-from-file-path"
    )

    parser.add_argument(
        "-t",
        "--do-not-delete",
        action="store_true",
        help="do not delete remote files missing from local folder"
    )

    parser.add_argument(
        "-o",
        "--allow-unknown",
        action="store_true",
        help="allow connection to unknown hosts"
    )

    parser.add_argument(
        "-r",
        "--create-remote-directory",
        action="store_true",
        help="Create remote base directory if missing on remote"
    )

    return parser
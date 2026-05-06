def _prompt_password(prompt="Password: "):
    """Prompt the user for password, printing asterisks for each character"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(fd)

    print(prompt, end="", flush=True)
    password = []
    try:
        while True:
            ch = sys.stdin.buffer.read(1)[0]
            if ch in (ord("\r"), ord("\n"), 4):  # If user presses Enter or ctrl-d
                print("\r")
                break
            elif ch == 127:  # DEL
                try:
                    password.pop()
                except IndexError:
                    pass
                else:
                    print("\b \b", end="", flush=True)
            elif ch == 3:  # ctrl-c
                print("^C", end="", flush=True)
                raise KeyboardInterrupt
            else:
                password.append(ch)
                print("*", end="", flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return bytes(password).decode()
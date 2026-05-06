def sync_readmes():
    """ just copies README.md into README for pypi documentation """
    print("syncing README")
    with open("README.md", 'r') as reader:
        file_text = reader.read()
    with open("README", 'w') as writer:
        writer.write(file_text)
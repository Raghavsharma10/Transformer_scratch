def find_censored_md5ext(post_id: int) -> Optional[str]:
    "Find MD5 for a censored post's ID, return None if can't find."
    try:
        last_pull_date = LAST_PULL_DATE_FILE.read_text().strip()
    except FileNotFoundError:
        last_pull_date = ""

    date = datetime.utcnow()
    date = f"{date.year}{date.month}{date.day}"

    if last_pull_date != date:
        update_batches()
        LAST_PULL_DATE_FILE.parent.mkdir(exist_ok=True, parents=True)
        LAST_PULL_DATE_FILE.write_text(date)

    # Faster than converting every ID in files to int
    post_id = str(post_id)

    for batch in BATCHES_DIR.iterdir():
        with open(batch, "r") as content:
            for line in content:
                an_id, its_md5_ext = line.split(":")

                if post_id == an_id:
                    return its_md5_ext.rstrip().split(".")

    return None
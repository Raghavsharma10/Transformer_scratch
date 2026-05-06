def update_batches() -> None:
    "Update id:md5.ext batches from Dencensooru's repository."
    batches_data  = requests.get(BATCHES_API_URL).json()
    batches_url   = {i["name"]: i["download_url"] for i in batches_data
                     if i["type"] == "file"}
    order_batches = sorted(batches_url, key=int)

    try:
        existing = set(os.listdir(BATCHES_DIR))
    except FileNotFoundError:
        BATCHES_DIR.mkdir(parents=True)
        existing = set()

    def get_batch(name: str) -> None:
        if name in existing and name != order_batches[-1]:
            return

        answer = requests.get(batches_url[name])
        try:
            answer.raise_for_status()
        except requests.RequestException:
            return

        with AtomicFile(BATCHES_DIR / name, "w") as file:
            file.write(answer.text)

    ThreadPool(8).map(get_batch, order_batches)
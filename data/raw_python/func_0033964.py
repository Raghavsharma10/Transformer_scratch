def demo():
    """Demonstrate progress bar."""
    from time import sleep
    maxProgress = 1000
    with ProgressBar(max=maxProgress) as progressbar:
        for i in range(-100, maxProgress):
            sleep(0.01)
            progressbar.update(i)
    progressbar2 = ProgressBar(max=maxProgress)
    for s in progressbar2.iterate(range(maxProgress)):
        sleep(0.01)
    for s in progressbar2.iterate(range(maxProgress), format='iteration %d'):
        sleep(0.01)
def color_parts(parts):
    """Adds colors to each part of the citation"""
    return parts._replace(
        title=Fore.GREEN + parts.title + Style.RESET_ALL,
        doi=Fore.CYAN + parts.doi + Style.RESET_ALL
    )
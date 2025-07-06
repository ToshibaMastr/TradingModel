import asyncio
from pathlib import Path

from klines.downloader import OHLCVDownloader, ProgressEvent
from rich.progress import BarColumn, Progress, TimeRemainingColumn

from .utils import set_seeds, user_data_init

user_data = Path("user_data")

set_seeds(2003)
user_data_init(user_data)


async def download(symbols: list[str], interval: str, start: int, end: int):
    downloader = OHLCVDownloader()

    with Progress(
        "[bold blue]{task.fields[pair]}[/]",
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeRemainingColumn(),
    ) as progress:
        tasks = {}

        for symbol in symbols:
            pair = symbol + "USDT"
            file = user_data / "data" / f"{pair}-{interval}.pkl"

            if file.exists():
                continue

            task_id = progress.add_task("", total=100, pair=pair)
            tasks[pair] = task_id

            def progevent(status: ProgressEvent, pair=pair):
                progress.update(tasks[pair], completed=status.percent)

            downloader.set_progress_hook(progevent)

            df = await downloader.download(pair, interval, start, end)
            df.to_pickle(file)


if __name__ == "__main__":
    asyncio.run(download(["BTC", "ETH"], "1m", 1581834371000, 1751834371000))

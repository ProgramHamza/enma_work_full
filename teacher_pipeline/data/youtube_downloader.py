from pathlib import Path
import sys
import csv
import traceback
from datetime import datetime


project_root = Path(__file__).resolve().parent
venv_site_packages = project_root / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

if venv_site_packages.exists():
    sys.path.insert(0, str(venv_site_packages))

import yt_dlp
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def format_duration(total_seconds: int) -> str:
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


link_files = sorted(
    path
    for path in project_root.iterdir()
    if path.is_file()
    and path.name != Path(__file__).name
    and path.name.startswith("video_list")
)

download_dir = project_root / "downloaded_videos"
download_dir.mkdir(parents=True, exist_ok=True)

log_csv = project_root / "video_list_log.csv"


def success_file_for(video_id: str):
    # any file that contains _{id} before extension
    matches = list(download_dir.glob(f"*_{video_id}.*"))
    return matches[0] if matches else None


def failed_marker_for(video_id: str):
    return download_dir / f"{video_id}.failed"


def is_already_handled(video_id: str) -> bool:
    if success_file_for(video_id):
        return True
    if failed_marker_for(video_id).exists():
        return True
    return False


all_links = []
for link_file in link_files:
    links = [line.strip() for line in link_file.read_text().splitlines() if line.strip()]
    all_links.extend((link_file.stem, link) for link in links)

print(f"Found {len(all_links)} video links in {len(link_files)} files.")

total_seconds = 0

metadata_opts = {
    "quiet": True,
    "skip_download": True,
}

download_opts = {
    "format": "best",
    "quiet": True,
    "outtmpl": str(download_dir / "%(extractor_key)s_%(id)s.%(ext)s"),
}


def append_csv_row(row: dict):
    header = ["timestamp", "source", "id", "url", "title", "duration", "filename", "status", "error"]
    exists = log_csv.exists()
    with log_csv.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


with yt_dlp.YoutubeDL(metadata_opts) as metadata_ydl, yt_dlp.YoutubeDL(download_opts) as download_ydl:
    if tqdm:
        pbar = tqdm(all_links, desc="videos", unit="vid")
    else:
        pbar = None

    iterator = enumerate(pbar if pbar is not None else all_links, 1)
    for idx, (source_name, video_link) in iterator:
        try:
            info = metadata_ydl.extract_info(video_link, download=False)
        except Exception as e:
            err = str(e)
            print(f"[{source_name}] metadata failed: {err}")
            append_csv_row({
                "timestamp": datetime.utcnow().isoformat(),
                "source": source_name,
                "id": "",
                "url": video_link,
                "title": "",
                "duration": 0,
                "filename": "",
                "status": "metadata_failed",
                "error": err,
            })
            continue

        if not info:
            print(f"[{source_name}] skipped: {video_link}")
            append_csv_row({
                "timestamp": datetime.utcnow().isoformat(),
                "source": source_name,
                "id": "",
                "url": video_link,
                "title": "",
                "duration": 0,
                "filename": "",
                "status": "skipped",
                "error": "no_info",
            })
            continue

        duration = int(info.get("duration") or 0)
        total_seconds += duration
        title = info.get("title") or ""
        if pbar:
            # update progress bar description
            try:
                pbar.set_description(f"{idx}/{len(all_links)} {title[:60]}")
                pbar.set_postfix({"id": info.get("id")})
            except Exception:
                pass
        else:
            print(f"[{source_name}] {duration:>5}s  {title}")

        video_id = str(info.get("id") or "")
        if video_id and is_already_handled(video_id):
            existing = success_file_for(video_id)
            status = "exists" if existing else "failed_marker"
            print(f"[{source_name}] already handled: {video_id} ({status})")
            append_csv_row({
                "timestamp": datetime.utcnow().isoformat(),
                "source": source_name,
                "id": video_id,
                "url": video_link,
                "title": title,
                "duration": duration,
                "filename": existing.name if existing else "",
                "status": status,
                "error": "",
            })
            continue

        try:
            download_ydl.download([video_link])
            # find downloaded file
            out = success_file_for(video_id)
            if out:
                print(f"[{source_name}] downloaded -> {out.name}")
                append_csv_row({
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": source_name,
                    "id": video_id,
                    "url": video_link,
                    "title": title,
                    "duration": duration,
                    "filename": out.name,
                    "status": "downloaded",
                    "error": "",
                })
            else:
                # no file found: mark failed
                msg = "downloaded_but_no_file"
                print(f"[{source_name}] download completed but no file found for {video_id}")
                failed_marker_for(video_id).write_text(msg)
                append_csv_row({
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": source_name,
                    "id": video_id,
                    "url": video_link,
                    "title": title,
                    "duration": duration,
                    "filename": "",
                    "status": "failed",
                    "error": msg,
                })
        except Exception as error:
            err_text = str(error)
            print(f"[{source_name}] download failed: {err_text}")
            # save a failed marker with traceback for later inspection
            tb = traceback.format_exc()
            try:
                failed_marker_for(video_id).write_text(tb)
            except Exception:
                pass
            append_csv_row({
                "timestamp": datetime.utcnow().isoformat(),
                "source": source_name,
                "id": video_id,
                "url": video_link,
                "title": title,
                "duration": duration,
                "filename": "",
                "status": "download_error",
                "error": err_text,
            })

print(f"Total duration: {format_duration(total_seconds)}")


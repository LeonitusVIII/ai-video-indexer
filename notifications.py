import datetime
import json
from pathlib import Path
from urllib import request
from urllib.error import URLError


def send_discord_webhook(webhook_url, title, message, color=5763719, fields=None):
    if not webhook_url or not webhook_url.strip():
        return False, "No Discord webhook configured."

    payload = {
        "embeds": [
            {
                "title": title,
                "description": message[:4096],
                "color": color,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "fields": fields or [],
            }
        ]
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        webhook_url.strip(),
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "VideoIndexer/1.0",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=15) as resp:
            if 200 <= resp.status < 300:
                return True, None
            return False, f"Discord returned HTTP {resp.status}"
    except URLError as e:
        return False, str(e.reason if hasattr(e, "reason") else e)


def notify_pipeline_event(webhook_url, event, folder_label, details, log_file=""):
    colors = {
        "complete": 5763719,
        "complete_with_failures": 16766720,
        "failed": 15548997,
        "stopped": 16776960,
    }
    titles = {
        "complete": "Video Indexer pipeline complete",
        "complete_with_failures": "Video Indexer pipeline finished with failures",
        "failed": "Video Indexer pipeline failed",
        "stopped": "Video Indexer pipeline stopped",
    }

    fields = [{"name": "Folder(s)", "value": folder_label[:1024], "inline": False}]
    if log_file:
        fields.append({"name": "Log file", "value": Path(log_file).name[:1024], "inline": False})

    return send_discord_webhook(
        webhook_url,
        titles.get(event, "Video Indexer pipeline update"),
        details[:4096],
        color=colors.get(event, 5763719),
        fields=fields,
    )

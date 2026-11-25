"""Run the remote probe or proxy with the local credential over SSH stdin."""

import argparse
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("operation", choices=["proxy", "probe", "replay"])
parser.add_argument("arm", choices=["baseline", "semantic", "visual", "dual", "current"])
parser.add_argument("batch", type=str)
parser.add_argument("--region", choices=["beijing", "singapore"], default="beijing")
args = parser.parse_args()
if not re.fullmatch("[A-Za-z0-9_-]+", args.batch):
    parser.error("Use an alphanumeric batch ID with optional hyphens and underscores")
env = (Path(__file__).resolve().parents[1] / ".env").read_text(encoding="utf-8-sig")
match = re.search(r"(?m)^\s*(?:export\s+)?DASHSCOPE_API_KEY\s*=\s*(.+)$", env)
if not match:
    parser.error("DASHSCOPE_API_KEY missing in workspace .env")
key = match[1].strip().strip("\"'")
if not key or any(ch.isspace() for ch in key):
    parser.error("Invalid credential format")
host = "dashscope.aliyuncs.com" if args.region == "beijing" else "dashscope-intl.aliyuncs.com"
cmd = (f"cd /home2/wanquan/VoLoAgent && DASHSCOPE_BASE_URL=https://{host}/compatible-mode/v1 "
       f"bash scripts/mem_libero.sh {args.operation} {args.arm} {args.batch}")
subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                "wanquan@222.20.97.104", cmd], input=(key + "\n").encode(), check=True)

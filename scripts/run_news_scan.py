import argparse
import json
import sys

from data.news_scanner import scan_slate_news

# One real news-scan pass over a slate's currently-eligible pool - see
# data/news_scanner.py's own module docstring for what this does and its
# real limitations (in particular: do not run this against a simulated/
# fictional slate and expect its findings to mean anything real).
#
# "Continuous... throughout the week" (the real design intent) means
# running THIS script repeatedly over the week, not a single run - this
# process exits after one pass, it does not loop or daemonize itself.
# Schedule it externally, e.g. a real cron entry:
#   0 */4 * * * cd /path/to/blank-app && venv/bin/python scripts/run_news_scan.py dk_thu_mon_2026_09_17
# (every 4 hours; tighten to hourly close to lock, per this project's own
# documented pre-lock-check cadence). Requires a real ANTHROPIC_API_KEY in
# the environment - see .env.example.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slate_id")
    args = parser.parse_args()

    try:
        result = scan_slate_news(args.slate_id)
    except RuntimeError as exc:
        print(f"News scan could not run: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2, default=str))
    if result["contradictions_found"] > 0:
        print(
            f"\n{result['contradictions_found']} real finding(s) contradict this tool's "
            "current gate - review get_contradictions() before locking any lineup.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

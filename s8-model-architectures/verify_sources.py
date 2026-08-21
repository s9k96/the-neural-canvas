"""Verify every arXiv-sourced date in sources.json against arXiv's own record.

The assignment's stated failure mode is an agent inventing a confident launch date. This
script exists so that no date on the timeline has to be taken on trust: it re-fetches each
arXiv entry and asserts that our claimed date equals the <published> field, which is the v1
submission timestamp.

    python s8-model-architectures/verify_sources.py

Exits 0 only if every checkable claim matches. Writes verification.log next to this file.
Needs network. Entries whose source_type is not 'arxiv' (a forum post, a model release) are
reported as EXEMPT with the reason, never silently skipped -- an unchecked claim that does
not appear in the output is indistinguishable from a claim that passed.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = HERE / "sources.json"
LOG = HERE / "verification.log"
API = "http://export.arxiv.org/api/query?id_list={ids}&max_results=100"
BATCH = 12

lines: list[str] = []


def say(msg: str = "") -> None:
    print(msg)
    lines.append(msg)


def fetch(ids: list[str]) -> str:
    url = API.format(ids=",".join(ids))
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                return r.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 3:
                raise SystemExit(f"FATAL: arXiv API unreachable after 4 attempts: {e}")
            time.sleep(2 * (attempt + 1))
    return ""


def parse(xml: str) -> dict[str, dict[str, str]]:
    """id -> {published, title}. arXiv returns <id>http://arxiv.org/abs/XXXX.YYYYYvN</id>."""
    out: dict[str, dict[str, str]] = {}
    for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        raw_id = re.search(r"<id>\s*(.*?)\s*</id>", entry, re.S)
        pub = re.search(r"<published>\s*(.*?)\s*</published>", entry, re.S)
        title = re.search(r"<title>\s*(.*?)\s*</title>", entry, re.S)
        if not (raw_id and pub):
            continue
        bare = raw_id.group(1).rsplit("/", 1)[-1]
        bare = re.sub(r"v\d+$", "", bare)
        out[bare] = {
            "published": pub.group(1),
            "title": re.sub(r"\s+", " ", title.group(1)).strip() if title else "",
        }
    return out


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def main() -> int:
    doc = json.loads(SOURCES.read_text(encoding="utf-8"))
    entries = doc["entries"]

    checkable = [e for e in entries if e.get("source_type") == "arxiv"]
    exempt = [e for e in entries if e.get("source_type") != "arxiv"]

    say("=" * 78)
    say("S8 chronology verification -- claimed dates vs arXiv <published> (v1 submission)")
    say("=" * 78)
    say(f"entries total {len(entries)}   arxiv-checkable {len(checkable)}   exempt {len(exempt)}")
    say()

    # --- ordering invariant: the file must already be in chronological order -----
    # Dates are not all the same precision. NTK-aware is known only to the month, so
    # comparing it against a day-precision neighbour as a raw string reports a false
    # conflict ("2023-06-27" > "2023-06"). Compare adjacent entries at the COARSER of
    # their two precisions: a month-precision entry is out of order only if its month
    # is genuinely wrong, not merely because it lacks a day.
    order_fail = []
    for a, b in zip(entries, entries[1:]):
        n = min(len(a["date"]), len(b["date"]))
        if a["date"][:n] > b["date"][:n]:
            order_fail.append(f"{a['id']} ({a['date']}) precedes {b['id']} ({b['date']})")

    # --- cross-check: the page's cards and the evidence file must agree ---------
    # mechanisms.js carries the prose; sources.json carries the citations. They are
    # separate files, so they can drift. Every card must correspond to an entry here,
    # every entry must have a card unless it is explicitly marked "card": false, and
    # the dates must be identical -- otherwise the page could display a date that was
    # never checked against arXiv, which is the whole failure this script exists to stop.
    card_fail = []
    js_path = HERE / "mechanisms.js"
    if js_path.exists():
        js = js_path.read_text(encoding="utf-8")
        cards = dict(re.findall(r"^\s{4}id: '([a-z0-9-]+)', date: '([\d-]+)'", js, re.M))
        by_id = {e["id"]: e for e in entries}
        for cid, cdate in cards.items():
            if cid not in by_id:
                card_fail.append(f"card {cid!r} in mechanisms.js has no entry in sources.json")
            elif by_id[cid]["date"] != cdate:
                card_fail.append(
                    f"card {cid!r} dated {cdate} but sources.json says {by_id[cid]['date']}")
        for e in entries:
            if e.get("card") is not False and e["id"] not in cards:
                card_fail.append(f"entry {e['id']!r} has no card in mechanisms.js")
        say(f"cross-check: {len(cards)} cards in mechanisms.js vs {len(entries)} evidence entries"
            f" -- {'OK' if not card_fail else str(len(card_fail)) + ' MISMATCH'}")
    else:
        card_fail.append("mechanisms.js not found -- cannot cross-check the page against evidence")
    say()

    # --- fetch -----------------------------------------------------------------
    fetched: dict[str, dict[str, str]] = {}
    ids = [e["arxiv_id"] for e in checkable]
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        say(f"  fetching {len(chunk)} records from arXiv ...")
        fetched.update(parse(fetch(chunk)))
        time.sleep(3)  # arXiv asks for ~1 request per 3s
    say()

    # --- compare ---------------------------------------------------------------
    date_fail, title_warn, missing = [], [], []
    say(f"{'id':<24} {'claimed':<12} {'arxiv v1':<12} {'':<4} arxiv_id")
    say("-" * 78)
    for e in checkable:
        aid, claimed = e["arxiv_id"], e["date"]
        rec = fetched.get(aid)
        if not rec:
            missing.append(aid)
            say(f"{e['id']:<24} {claimed:<12} {'NOT FOUND':<12} FAIL {aid}")
            continue
        actual = rec["published"][:10]
        ok = actual == claimed
        if not ok:
            date_fail.append(f"{e['id']}: claimed {claimed}, arXiv says {actual} ({aid})")
        if rec["title"] and norm(e["title"])[:45] not in norm(rec["title"]):
            title_warn.append(f"{e['id']}: title drift -- ours {e['title']!r} vs arXiv {rec['title']!r}")
        say(f"{e['id']:<24} {claimed:<12} {actual:<12} {'ok' if ok else 'FAIL':<4} {aid}")

    # --- exempt ----------------------------------------------------------------
    say()
    say("Exempt from the arXiv check (stated, not skipped):")
    say("-" * 78)
    for e in exempt:
        prec = e.get("date_precision", "day")
        reason = e.get("caveat", "non-arXiv source")
        say(f"  {e['id']:<22} {e['date']:<10} [{e['source_type']}, {prec} precision]")
        say(f"    {reason}")

    # --- verdict ---------------------------------------------------------------
    say()
    say("=" * 78)
    for w in title_warn:
        say(f"WARN  {w}")
    for f in order_fail:
        say(f"FAIL  out of chronological order: {f}")
    for f in date_fail:
        say(f"FAIL  {f}")
    for m in missing:
        say(f"FAIL  arXiv returned no record for {m}")

    for f in card_fail:
        say(f"FAIL  {f}")

    bad = len(date_fail) + len(order_fail) + len(missing) + len(card_fail)
    if bad:
        say(f"RESULT: FAIL -- {bad} problem(s). The timeline must not ship with these.")
    else:
        say(f"RESULT: PASS -- {len(checkable)}/{len(checkable)} arXiv dates match v1 submission,")
        say(f"        {len(entries)} entries in correct chronological order,")
        say(f"        {len(exempt)} non-arXiv claim(s) declared with reasons above,")
        say(f"        page cards and evidence entries in agreement.")
    say("=" * 78)

    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {LOG.relative_to(HERE.parent)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

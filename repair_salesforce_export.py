"""
Repair SalesforceLatest.csv records mangled by an extended-ASCII export bug.

The export occasionally collapses the byte sequence `<char>","` (closing
quote + comma + opening quote between two quoted fields) into a single
non-ASCII byte followed by `"`, eating the field separator. The CSV
parser then sees the next field run on without a separator, the row's
field count is wrong, and downstream filters drop the record.

Repair: at every position where a non-ASCII byte is immediately
followed by `"` and the next byte is NOT a field separator
(`,`, `\\r`, `\\n`) or EOF, insert `,"` after the `"`. This restores
a well-formed CSV row. The accented name remains mangled — the
underlying character is lost — but the transaction is preserved.

Run: python repair_salesforce_export.py
Output: SalesforceLatest.csv (overwritten, original moved to .bak)
"""
import os
import shutil
import sys

WORKING_DIR = "/Users/antho/Documents/WPI-MW"
os.chdir(WORKING_DIR)

SRC = "SalesforceLatest.csv"
BAK = "SalesforceLatest.csv.bak"


def find_breaks(data: bytes):
    """Yield indices i where data[i] is a non-ASCII byte, data[i+1] is `"`,
    and data[i+2] is not a CSV separator. These are the broken positions."""
    n = len(data)
    seps = {ord(","), ord("\r"), ord("\n")}
    quote = ord('"')
    for i in range(n - 2):
        b = data[i]
        if b < 0x80:
            continue
        if data[i + 1] != quote:
            continue
        if data[i + 2] in seps:
            continue
        yield i


def line_at(data: bytes, pos: int):
    start = data.rfind(b"\n", 0, pos) + 1
    end = data.find(b"\n", pos)
    if end == -1:
        end = len(data)
    return start, end, data[start:end]


def event_name_in_line(line: bytes):
    """First field of each row is the event name (quoted)."""
    if not line.startswith(b'"'):
        return None
    close = line.find(b'"', 1)
    if close == -1:
        return None
    return line[1:close].decode("latin1", errors="replace")


def main():
    if not os.path.exists(SRC):
        print(f"ERROR: {SRC} not found", file=sys.stderr)
        sys.exit(1)

    with open(SRC, "rb") as f:
        data = f.read()
    print(f"Read {len(data):,} bytes from {SRC}")

    breaks = list(find_breaks(data))
    print(f"Found {len(breaks)} broken field boundaries")

    if not breaks:
        print("Nothing to repair.")
        return

    # Affected events
    events = {}
    for pos in breaks:
        _, _, line = line_at(data, pos)
        ev = event_name_in_line(line) or "(unknown)"
        events[ev] = events.get(ev, 0) + 1
    print("\nAffected events:")
    for ev, n in sorted(events.items(), key=lambda x: -x[1]):
        print(f"  {n:4d}  {ev}")

    # Build repaired buffer: at each break, insert `,"` after the `"`.
    pieces = []
    last = 0
    for pos in breaks:
        # Copy through the closing quote
        pieces.append(data[last:pos + 2])
        pieces.append(b',"')
        last = pos + 2
    pieces.append(data[last:])
    fixed = b"".join(pieces)
    print(f"\nRepaired size: {len(fixed):,} bytes (Δ +{len(fixed) - len(data):,})")

    # Back up original, then write fix in place.
    if not os.path.exists(BAK):
        shutil.copy2(SRC, BAK)
        print(f"Backup written: {BAK}")
    else:
        print(f"Backup already exists at {BAK} (not overwriting)")

    with open(SRC, "wb") as f:
        f.write(fixed)
    print(f"Repaired {SRC} written in place")


if __name__ == "__main__":
    main()

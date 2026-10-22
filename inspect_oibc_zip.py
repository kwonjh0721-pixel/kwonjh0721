"""Create a small schema/sample report without extracting the competition ZIP.

Usage: python inspect_oibc_zip.py OIBC_2026_DATA.zip --output oibc_report.json
Python 3.10+; no third-party dependencies. This does not train a model.
The report includes sample data: review it before sharing.
"""

import argparse
import csv
import io
import json
from pathlib import Path
import zipfile


def inspect_archive(path: Path, rows: int) -> dict:
    report = {"archive": path.name, "files": []}
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            item = {"name": member.filename, "bytes": member.file_size}
            report["files"].append(item)
            suffix = Path(member.filename).suffix.lower()
            if suffix not in {".csv", ".tsv", ".txt"}:
                item["status"] = "listed_only"
                continue
            if member.flag_bits & 1:
                item["status"] = "encrypted"
                continue
            try:
                # Bound decompression and report size even for large input files.
                with archive.open(member) as stream:
                    raw = stream.read(256 * 1024)
                truncated = len(raw) < member.file_size
                if truncated:
                    # Drop the last partial line, including any partial character.
                    boundary = raw.rfind(b"\n")
                    if boundary < 0:
                        item["status"] = "no_complete_line_in_preview"
                        continue
                    raw = raw[:boundary + 1]
                for encoding in ("utf-8-sig", "cp949", "utf-16"):
                    try:
                        content = raw.decode(encoding)
                        break
                    except UnicodeError:
                        continue
                else:
                    item["status"] = "unrecognized_encoding"
                    continue
                item.update(encoding=encoding, preview_truncated=truncated)
                if suffix == ".txt":
                    item["text_preview"] = content[:4000]
                    item["status"] = "text_preview_only"
                    continue
                try:
                    dialect = csv.Sniffer().sniff(content[:8192], delimiters=",\t;|")
                    delimiter = dialect.delimiter
                except csv.Error:
                    delimiter = "\t" if suffix == ".tsv" else ","
                reader = csv.reader(io.StringIO(content), delimiter=delimiter)
                item["delimiter"] = delimiter
                item["first_row"] = next(reader, [])
                item["sample_rows"] = []
                for _ in range(rows):
                    row = next(reader, None)
                    if row is None:
                        break
                    item["sample_rows"].append([cell[:300] for cell in row])
                item["status"] = "preview_only_header_not_inferred"
            except (OSError, ValueError, RuntimeError, csv.Error, zipfile.BadZipFile) as error:
                item["status"] = "preview_error"
                item["error"] = str(error)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("oibc_report.json"))
    parser.add_argument("--rows", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.rows <= 100:
        parser.error("--rows must be between 1 and 100")
    report = inspect_archive(args.archive, args.rows)
    # Exclusive creation protects existing user files, including the input archive.
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}: {len(report['files'])} files; previews only.")


if __name__ == "__main__":
    main()

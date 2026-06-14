import hashlib
import shutil
from pathlib import Path

from core.workspace import SoulDriveWorkspace


def import_paper_into_workspace(workspace: SoulDriveWorkspace, source_path: str) -> dict:
    source = Path(source_path)
    display_name = source.name or "unknown"

    if source.suffix.lower() != ".pdf":
        return {
            "name": display_name,
            "status": "rejected",
            "error_code": "UNSUPPORTED_FILE_TYPE",
        }
    if not source.exists() or not source.is_file():
        return {
            "name": display_name,
            "status": "rejected",
            "error_code": "SOURCE_NOT_FOUND",
        }

    papers_dir = Path(workspace.papers_path)
    papers_dir.mkdir(parents=True, exist_ok=True)
    base_destination = papers_dir / safe_pdf_filename(source.name)

    try:
        if source.resolve() == base_destination.resolve():
            return {
                "name": base_destination.name,
                "status": "already_present",
            }
    except OSError:
        pass

    if base_destination.exists() and same_file_content(source, base_destination):
        return {
            "name": base_destination.name,
            "status": "already_present",
        }

    destination = available_paper_path(base_destination)
    shutil.copy2(str(source), str(destination))
    return {
        "name": destination.name,
        "status": "imported",
    }


def import_drive_root_papers(drive_path: str, workspace: SoulDriveWorkspace) -> list[dict]:
    root = Path(drive_path)
    if not root.exists():
        return []
    return [
        import_paper_into_workspace(workspace, str(path))
        for path in sorted(root.glob("*.pdf"), key=lambda item: item.name.lower())
    ]


def safe_pdf_filename(filename: str) -> str:
    cleaned = "".join(
        "_" if character in '<>:"/\\|?*' or ord(character) < 32 else character
        for character in Path(filename).name
    ).strip(" .")
    if not cleaned:
        cleaned = "paper.pdf"
    path = Path(cleaned)
    if path.suffix.lower() != ".pdf":
        cleaned = f"{path.stem or 'paper'}.pdf"
    return cleaned


def available_paper_path(destination: Path) -> Path:
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    for index in range(2, 1000):
        candidate = destination.with_name(f"{stem} ({index}){suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"too many duplicate paper names for {destination.name}")


def same_file_content(left: Path, right: Path) -> bool:
    try:
        if left.stat().st_size != right.stat().st_size:
            return False
    except OSError:
        return False
    return file_sha256(left) == file_sha256(right)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

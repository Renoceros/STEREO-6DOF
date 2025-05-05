import os
from collections import defaultdict

MAX_FILES_PER_TYPE = 3
OUTPUT_FILE = "STRUCT.txt"
DONTCRAWL = {"depricated","video/preprocessed","venv",".git"}  # <- folders to skip (add more if needed)
EXT_ALWAYS_SHOW = {".py", ".ipynb"}

def format_dir_tree(root_dir):
    lines = []

    def walk(dir_path, indent=""):
        folder_name = os.path.basename(dir_path)
        if folder_name in DONTCRAWL:
            lines.append(f"{indent}{folder_name}/ [skipped]")
            return

        lines.append(f"{indent}{folder_name}/")
        indent += "    "

        files_by_ext = defaultdict(list)
        try:
            entries = sorted(os.listdir(dir_path))
        except PermissionError:
            lines.append(f"{indent}<Permission Denied>")
            return

        for entry in entries:
            full_path = os.path.join(dir_path, entry)
            if os.path.isdir(full_path):
                walk(full_path, indent)
            else:
                ext = os.path.splitext(entry)[-1].lower()
                files_by_ext[ext].append(entry)

        for ext, files in files_by_ext.items():
            if ext in EXT_ALWAYS_SHOW or len(files) <= MAX_FILES_PER_TYPE:
                for f in files:
                    lines.append(f"{indent}- {f}")
            else:
                for f in files[:MAX_FILES_PER_TYPE]:
                    lines.append(f"{indent}- {f}")
                lines.append(f"{indent}- ... ({len(files) - MAX_FILES_PER_TYPE} more .{ext[1:]})")

    walk(root_dir)
    return "\n".join(lines)

if __name__ == "__main__":
    root_directory = "."  # Change if needed
    structure = format_dir_tree(root_directory)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(structure)
    print(f"Structure saved to {OUTPUT_FILE}")

import shutil
import subprocess
import sys
import os
import argparse

def run_pipreqs(project_dir: str):
    if not shutil.which("pipreqs"):
        print("pipreqs not found. Install it: pip install pipreqs")
        sys.exit(1)
    subprocess.run(["pipreqs", project_dir, "--force"], check=True)

def read_requirements(req_path: str):
    if not os.path.exists(req_path):
        return []
    with open(req_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.startswith("#")]
    return lines

def write_env_yaml(out_path: str, name: str, python_version: str, reqs: list, channels: list):
    lines = []
    lines.append(f"name: {name}")
    lines.append("channels:")
    for c in channels:
        lines.append(f"  - {c}")
    lines.append("dependencies:")
    lines.append(f"  - python={python_version}")
    lines.append(f"  - pip")
    lines.append(f"  - pip:")
    if reqs:
        for r in reqs:
            lines.append(f"    - {r}")
    else:
        lines.append("    -")  # empty pip block to avoid syntax error
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Written {out_path}")

def main():
    p = argparse.ArgumentParser(description="Generate environment.yaml from project imports")
    p.add_argument("--project", "-p", default=".", help="Project root to scan for imports")
    p.add_argument("--out", "-o", default="environment.yaml", help="Output env yaml path")
    p.add_argument("--name", default="project-env", help="Conda environment name")
    p.add_argument("--python", default="3.10", help="Python version for env")
    p.add_argument("--channels", nargs="+", default=["defaults", "conda-forge"], help="Conda channels")
    args = p.parse_args()

    project_dir = os.path.abspath(args.project)
    req_path = os.path.join(project_dir, "requirements.txt")

    reqs = read_requirements(req_path)
    write_env_yaml(os.path.abspath(args.out), args.name, args.python, reqs, args.channels)

if __name__ == "__main__":
    main()
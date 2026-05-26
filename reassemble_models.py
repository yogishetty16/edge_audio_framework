#!/usr/bin/env python3
"""
reassemble_models.py
====================
Utility script to reassemble split zip parts, validate integrity,
extract models, and place them into the models/ directory.
"""

import os
import re
import sys
import shutil
import zipfile
from pathlib import Path

def force_rmtree(path):
    import stat
    path = Path(path)
    if not path.exists():
        return
    def remove_readonly(func, p, excinfo):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass
    shutil.rmtree(path, onerror=remove_readonly)

def main():
    print("  ================================================")
    print("    Edge Audio Framework — Reassemble Models")
    print("  ================================================")
    print()

    base_dir = Path(__file__).resolve().parent
    zip_path = base_dir / "models.zip"
    temp_dir = base_dir / "models_temp"
    target_dir = base_dir / "models"

    try:
        # ----------------------------------------------------------------------
        # [1/5] Scanning for parts
        # ----------------------------------------------------------------------
        print("  [1/5] Scanning for parts...")
        part_files = list(base_dir.glob("models.zip.part*"))
        
        if not part_files:
            print("    [ERROR] No split zip parts found in the project root.")
            print("            Expected files like models.zip.part1, models.zip.part2, etc.")
            sys.exit(1)

        # Sort parts numerically
        def get_part_num(path):
            match = re.search(r'\.part(\d+)$', path.name)
            return int(match.group(1)) if match else 999999

        part_files.sort(key=get_part_num)

        # Validate each part
        total_bytes = 0
        for i, part in enumerate(part_files, 1):
            expected_name = f"models.zip.part{i}"
            if part.name != expected_name:
                print(f"    [ERROR] Missing part file or parts out of order.")
                print(f"            Expected: {expected_name}, but found: {part.name}")
                sys.exit(1)
            
            if not part.exists():
                print(f"    [ERROR] Part {part.name} does not exist.")
                sys.exit(1)
            
            size_bytes = part.stat().st_size
            if size_bytes == 0:
                print(f"    [ERROR] Part {part.name} is empty (0 bytes).")
                sys.exit(1)
            
            size_mb = size_bytes / (1024 * 1024)
            print(f"    {part.name}  ({size_mb:.1f} MB)")
            total_bytes += size_bytes

        total_mb = total_bytes / (1024 * 1024)
        print(f"    Found {len(part_files)} parts. Total: {total_mb:.1f} MB")
        print()

        # ----------------------------------------------------------------------
        # [2/5] Joining parts
        # ----------------------------------------------------------------------
        print("  [2/5] Joining parts into models.zip...")
        with open(zip_path, "wb") as outfile:
            for part in part_files:
                print(f"    Reading {part.name}...")
                with open(part, "rb") as infile:
                    # Read and write in chunks to avoid high memory usage
                    chunk_size = 16 * 1024 * 1024 # 16 MB chunks
                    while True:
                        chunk = infile.read(chunk_size)
                        if not chunk:
                            break
                        outfile.write(chunk)
        
        assembled_size = zip_path.stat().st_size
        assembled_size_mb = assembled_size / (1024 * 1024)
        print(f"    [OK] models.zip assembled ({assembled_size_mb:.1f} MB)")
        print()

        # ----------------------------------------------------------------------
        # [3/5] Validating zip
        # ----------------------------------------------------------------------
        print("  [3/5] Validating zip integrity...")
        if not zipfile.is_zipfile(zip_path):
            print("    [ERROR] models.zip is corrupted — parts may be incomplete or downloaded incorrectly")
            sys.exit(1)

        with zipfile.ZipFile(zip_path) as zf:
            namelist = zf.namelist()
            file_count = len(namelist)
            print(f"    [OK] Valid zip file containing {file_count} files")
        print()

        # ----------------------------------------------------------------------
        # Safety Check before extraction
        # ----------------------------------------------------------------------
        if target_dir.exists():
            print("  [WARN] models/ folder already exists.")
            print("         This will overwrite existing model files.")
            try:
                input("         Press Enter to continue or Ctrl+C to cancel.")
            except (KeyboardInterrupt, EOFError):
                print("\n  Operation cancelled by user.")
                sys.exit(1)
            print()

        # ----------------------------------------------------------------------
        # [4/5] Extracting
        # ----------------------------------------------------------------------
        print("  [4/5] Extracting...")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                print(f"    Extracting {name}")
                zf.extract(name, path=temp_dir)
        
        print(f"    [OK] {file_count} files extracted")
        print()

        # ----------------------------------------------------------------------
        # [5/5] Arranging into models/
        # ----------------------------------------------------------------------
        print("  [5/5] Arranging into models/ folder...")
        
        # Determine extraction directory structure
        extract_root = temp_dir
        children = [c for c in extract_root.iterdir()]
        
        # CASE C: single nested wrapper folder
        if len(children) == 1 and children[0].is_dir():
            nested_dir = children[0]
            nested_children = list(nested_dir.iterdir())
            nested_dirs = [nc for nc in nested_children if nc.is_dir()]
            nested_files = [nc for nc in nested_children if nc.is_file()]
            has_models_subfolder = any(nc.name == "models" for nc in nested_dirs)
            
            if has_models_subfolder or len(nested_dirs) > 1 or (len(nested_dirs) == 1 and len(nested_files) == 0):
                extract_root = nested_dir
                children = nested_children

        # CASE A: models subfolder exists
        models_subfolder = next((c for c in children if c.is_dir() and c.name == "models"), None)
        
        # To avoid leaving a half-extracted or corrupted target directory,
        # we stage the files into models_new first, then perform a quick rename swap.
        models_new = base_dir / "models_new"
        if models_new.exists():
            force_rmtree(models_new)
        models_new.mkdir(parents=True, exist_ok=True)
        
        if models_subfolder:
            # Move contents of models_subfolder to models_new
            for item in models_subfolder.iterdir():
                shutil.move(str(item), str(models_new / item.name))
        else:
            # CASE B: move subfolders/files from extract_root to models_new
            for item in children:
                shutil.move(str(item), str(models_new / item.name))
                
        # Now swap models_new with target_dir
        if target_dir.exists():
            backup_dir = base_dir / "models_backup"
            if backup_dir.exists():
                force_rmtree(backup_dir)
            shutil.move(str(target_dir), str(backup_dir))
            try:
                shutil.move(str(models_new), str(target_dir))
                force_rmtree(backup_dir)
            except Exception as swap_err:
                # If swap fails, try to restore target_dir from backup
                if backup_dir.exists() and not target_dir.exists():
                    shutil.move(str(backup_dir), str(target_dir))
                raise swap_err
        else:
            shutil.move(str(models_new), str(target_dir))

        # Verify models folder top-level directories
        model_dirs = [d for d in target_dir.iterdir() if d.is_dir()]
        print(f"    [OK] models/ folder ready with {len(model_dirs)} model directories:")
        for md in model_dirs:
            print(f"      - {md.name}")
        print()

    except SystemExit as se:
        # sys.exit(1) was called, propagate it so it triggers finally
        raise se
    except Exception as e:
        print(f"\n  [ERROR] Execution failed: {e}")
        print("          Please check if your disk space is sufficient, file permissions are correct,")
        print("          and all model parts have been fully downloaded.")
        sys.exit(1)

    finally:
        # Cleanup
        cleaned = []
        for path, name in [
            (temp_dir, "models_temp/"),
            (zip_path, "models.zip"),
            (base_dir / "models_new", "models_new/"),
            (base_dir / "models_backup", "models_backup/")
        ]:
            if path.exists():
                try:
                    if path.is_dir():
                        force_rmtree(path)
                    else:
                        path.unlink()
                    if name in ["models_temp/", "models.zip"]:
                        cleaned.append(name)
                except Exception:
                    pass
        
        if cleaned:
            for item in cleaned:
                print(f"  [CLEANUP] Removed {item}")

    print()
    print("  ================================================")
    print("    Done. models/ is ready.")
    print("    Next step: python fast_run.py --calibrate --duration 10")
    print("  ================================================")

if __name__ == "__main__":
    main()

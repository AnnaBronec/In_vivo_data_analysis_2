# analysis_wrapper.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import runpy

# Pfad zu deinem bestehenden Script:
MAIN_FILE = Path(__file__).parent / "Main_safe.py"

def main_safe(base_path: str, lfp_filename: str):
    """
    Wrapper: Setzt BASE_PATH/LFP_FILENAME und führt Main_safe.py
    in diesem Kontext aus (ohne das Script umzuschreiben).
    """
    init_globals = {
        "BASE_PATH": str(Path(base_path).expanduser().resolve()),
        "LFP_FILENAME": str(lfp_filename),
    }
    # Script ausführen – es nutzt die gesetzten Globals
    runpy.run_path(str(MAIN_FILE), init_globals=init_globals)

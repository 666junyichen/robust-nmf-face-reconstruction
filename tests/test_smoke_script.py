import subprocess
import sys
from pathlib import Path


def test_smoke_experiment_script_reports_both_methods():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "smoke_experiment.py")],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "L2-NMF" in completed.stdout
    assert "L21-NMF" in completed.stdout
    assert "PASS" in completed.stdout

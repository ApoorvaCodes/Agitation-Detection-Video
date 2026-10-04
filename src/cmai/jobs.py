"""Isolated local analysis process; cancellation stops decoding/model work."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

from cmai.bundle import ROOT
from person1.contracts import Person1VideoResult
from person2.contracts import Person2VideoResult


class AnalysisJob:
    def __init__(self, path, bundle, config):
        self.directory = Path(path).parent
        self.status_path = self.directory / "progress.json"
        for name in ("progress.json", "perception.json", "candidates.json"):
            (self.directory / name).unlink(missing_ok=True)
        request = dict(path=str(path), p1_configuration=asdict(config), bundle=bundle.metadata.model_dump(mode="json"),
                       bank=bundle.bank.model_dump(mode="json") if bundle.bank else None)
        request_path = self.directory / "request.json"
        request_path.write_text(json.dumps(request))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        # Worker does local perception only; no remote verifier credentials needed.
        for name in ("GROQ_API_KEY", "SUPABASE_KEY", "SUPABASE_SERVICE_ROLE_KEY"):
            env.pop(name, None)
        self.log = open(self.directory / "analysis.log", "w")
        self.process = subprocess.Popen([sys.executable, "-m", "cmai.worker", str(request_path)],
                                         env=env, stdout=self.log, stderr=self.log)
        self.cancelled = False

    def status(self):
        if self.cancelled:
            return {"phase": "cancelled", "message": "Analysis cancelled; partial results discarded."}
        if self.status_path.exists():
            status = json.loads(self.status_path.read_text())
        else:
            status = {"phase": "running", "message": "Starting video validation…", "progress": 0}
        code = self.process.poll()
        if code is not None:
            self.log.close()
            if status["phase"] not in {"complete", "error"}:
                status = {"phase": "error", "message": f"Analysis process exited ({code}). See local analysis.log for dependency/native errors."}
        return status

    def cancel(self):
        self.cancelled = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        self.log.close()

    def results(self):
        if self.status()["phase"] != "complete":
            raise ValueError("analysis did not complete")
        return (Person1VideoResult.model_validate_json((self.directory / "perception.json").read_text()),
                Person2VideoResult.model_validate_json((self.directory / "candidates.json").read_text()))


def start_analysis(path, bundle, config):
    return AnalysisJob(path, bundle, config)

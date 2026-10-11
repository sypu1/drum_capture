"""The graphical live view: a small local web page.

Served by app.py next to the Gradio UI (http://127.0.0.1:7860/live).  The audio
is captured by Python (sounddevice), not by the browser, so the delay is only
the model's settle time.  Hits travel to the page as server-sent events.
"""
from __future__ import annotations

import json
import queue
import tempfile
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .audio_io import AudioError
from .backends import all_backends, get_backend
from .config import DRUM_ORDER, DRUMS, GRIDS
from .live import FileSource, LiveSession, MicSource, list_input_devices

router = APIRouter()
_lock = threading.Lock()
_state: dict = {"session": None, "dir": None, "take": None}


class StartRequest(BaseModel):
    model: str = "builtin"
    device: str | None = None
    simulate: str | None = None       # path of a file to play through the view instead of the mic
    sensitivity: float = 1.0
    gate_db: float = -45.0


class ChartRequest(BaseModel):
    model: str = "builtin"
    title: str = "Live take"
    grid: str = "16"
    style: str = "both"


@router.get("/live", response_class=HTMLResponse)
def page():
    return (Path(__file__).parent / "live_page.html").read_text(encoding="utf-8")


@router.get("/live/api/info")
def info():
    models = []
    for key, b in all_backends().items():
        ok, hint = b.available()
        models.append(dict(key=key, label=b.label, available=ok, hint=hint, description=b.description))
    s = _state["session"]
    return dict(models=models, devices=list_input_devices(), grids=list(GRIDS),
                drums=[dict(key=d, label=DRUMS[d]["label"], short=DRUMS[d]["short"]) for d in DRUM_ORDER],
                running=bool(s and s.running), has_take=bool(_state["take"]))


@router.post("/live/api/start")
def start(req: StartRequest):
    with _lock:
        if _state["session"] and _state["session"].running:
            raise HTTPException(409, "A live session is already running.")
        try:
            backend = get_backend(req.model)
            if req.simulate:
                if not Path(req.simulate).expanduser().is_file():
                    raise AudioError(f"File not found: {req.simulate}")
                source = FileSource(Path(req.simulate).expanduser())
            else:
                dev = req.device
                source = MicSource(int(dev) if dev and dev.isdigit() else (dev or None))
        except AudioError as exc:
            raise HTTPException(400, str(exc))
        _state["dir"] = Path(tempfile.mkdtemp(prefix="drumsheet_live_"))
        _state["take"] = None
        _state["session"] = LiveSession(source, backend, sensitivity=req.sensitivity,
                                        gate_db=req.gate_db).start()
    return dict(ok=True)


@router.get("/live/api/events")
def events():
    """Server-sent events: one JSON object per hit / level update."""
    session = _state["session"]
    if session is None:
        raise HTTPException(404, "No session.")
    q = session.listen()
    # a page that connects late still gets everything played so far
    backlog = [dict(type="hit", t=h.time, drum=h.drum, vel=h.velocity) for h in list(session.hits)]

    def stream():
        for e in backlog:
            yield f"data: {json.dumps(e)}\n\n"
        while True:
            try:
                e = q.get(timeout=1.0)
            except queue.Empty:
                if not session.running:
                    yield 'data: {"type": "end"}\n\n'
                    return
                yield ": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(e)}\n\n"
            if e["type"] == "end":
                return

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/live/api/stop")
def stop():
    with _lock:
        session = _state["session"]
        if session is None:
            raise HTTPException(404, "No session.")
        session.stop()
        try:
            _state["take"] = session.save(_state["dir"] / f"live_take_{time.strftime('%H%M%S')}.wav")
        except AudioError as exc:
            raise HTTPException(400, str(exc))
    return dict(ok=True, seconds=round(session.engine.now, 1), hits=len(session.hits),
                take=f"/live/files/{_state['take'].name}")


@router.post("/live/api/chart")
def chart(req: ChartRequest):
    """Make PDF / MIDI / MusicXML from the take that was just recorded."""
    from .pipeline import Options, run
    if not _state["take"]:
        raise HTTPException(404, "Record a take first.")
    try:
        result = run(_state["take"], Options(out_dir=_state["dir"], model=req.model, separate="no",
                                             title=req.title or "Live take", grid=req.grid,
                                             style=req.style), log=lambda *a: None)
    except AudioError as exc:
        raise HTTPException(400, str(exc))
    return dict(summary=result.summary,
                files=[dict(kind=k, name=p.name, url=f"/live/files/{p.name}") for k, p in result.files.items()])


@router.get("/live/files/{name}")
def files(name: str):
    folder = _state["dir"]
    path = (folder / name) if folder else None
    if not path or path.parent != folder or not path.is_file():
        raise HTTPException(404, "No such file.")
    return FileResponse(path, filename=name)

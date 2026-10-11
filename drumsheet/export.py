"""MIDI and MusicXML export of a quantised Score."""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from .config import DRUM_ORDER, DRUMS
from .rhythm import DIV, STRAIGHT, TRIPLET, Score, beat_events, slot_time


def export_midi(score: Score, path: str | Path) -> Path:
    """General-MIDI drum track (channel 10) in strict time at the detected tempo."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=score.bpm)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(score.beats_per_bar, 4, 0))
    kit = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    for b, bar in enumerate(score.bars):
        for slot, drums in bar.items():
            t = slot_time(score, b, slot)
            for drum, vel in drums.items():
                kit.notes.append(pretty_midi.Note(velocity=int(vel), pitch=DRUMS[drum]["midi"],
                                                  start=t, end=t + 0.1))
    pm.instruments.append(kit)
    pm.write(str(Path(path)))
    return Path(path)


def export_musicxml(score: Score, path: str | Path) -> Path:
    """MusicXML 3.1 percussion part.  Opens in MuseScore, Sibelius, Finale, Dorico..."""
    table = TRIPLET if score.sub in (3, 6) else STRAIGHT
    used = [d for d in DRUM_ORDER if d in score.used_drums()] or ["snare"]
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
           '"http://www.musicxml.org/dtds/partwise.dtd">',
           '<score-partwise version="3.1">',
           f'  <work><work-title>{escape(score.title)}</work-title></work>',
           '  <identification><encoding><software>drumsheet</software></encoding></identification>',
           '  <part-list>', '    <score-part id="P1">', '      <part-name>Drumset</part-name>']
    for d in used:
        out.append(f'      <score-instrument id="P1-I{DRUMS[d]["midi"] + 1}">'
                   f'<instrument-name>{DRUMS[d]["label"]}</instrument-name></score-instrument>')
    for d in used:
        m = DRUMS[d]["midi"] + 1
        out.append(f'      <midi-instrument id="P1-I{m}"><midi-channel>10</midi-channel>'
                   f'<midi-program>1</midi-program><midi-unpitched>{m}</midi-unpitched></midi-instrument>')
    out += ['    </score-part>', '  </part-list>', '  <part id="P1">']

    def note_xml(drum, dur, chord):
        kind, dots, triplet = table[dur]
        step, octave = DRUMS[drum]["xml"]
        head = ("circle-x" if drum == "hihat_open" else "x") if DRUMS[drum]["head"] == "x" else "normal"
        return ("      <note>" + ("<chord/>" if chord else "") +
                f"<unpitched><display-step>{step}</display-step>"
                f"<display-octave>{octave}</display-octave></unpitched>"
                f"<duration>{dur}</duration><instrument id=\"P1-I{DRUMS[drum]['midi'] + 1}\"/>"
                f"<voice>1</voice><type>{kind}</type>" + "<dot/>" * dots +
                ("<time-modification><actual-notes>3</actual-notes>"
                 "<normal-notes>2</normal-notes></time-modification>" if triplet else "") +
                f"<stem>up</stem><notehead>{head}</notehead></note>")

    def rest_xml(dur):
        kind, dots, triplet = table[dur]
        return (f"      <note><rest/><duration>{dur}</duration><voice>1</voice>"
                f"<type>{kind}</type>" + "<dot/>" * dots +
                ("<time-modification><actual-notes>3</actual-notes>"
                 "<normal-notes>2</normal-notes></time-modification>" if triplet else "") + "</note>")

    for b, bar in enumerate(score.bars or [{}]):
        out.append(f'    <measure number="{b + 1}">')
        if b == 0:
            out += ['      <attributes>', f'        <divisions>{DIV}</divisions>',
                    '        <key><fifths>0</fifths></key>',
                    f'        <time><beats>{score.beats_per_bar}</beats><beat-type>4</beat-type></time>',
                    '        <clef><sign>percussion</sign><line>2</line></clef>',
                    '      </attributes>',
                    '      <direction placement="above"><direction-type><metronome>'
                    f'<beat-unit>quarter</beat-unit><per-minute>{score.bpm:.0f}</per-minute>'
                    f'</metronome></direction-type><sound tempo="{score.bpm:.1f}"/></direction>']
        if not bar:
            out.append(f'      <note><rest measure="yes"/><duration>{DIV * score.beats_per_bar}'
                       '</duration><voice>1</voice></note>')
        else:
            for beat in range(score.beats_per_bar):
                for _, drums, dur in beat_events(score, b, beat):
                    if not drums:
                        out.append(rest_xml(dur))
                        continue
                    ordered = sorted(drums, key=lambda d: DRUMS[d]["staff_step"])
                    for i, d in enumerate(ordered):
                        out.append(note_xml(d, dur, chord=i > 0))
        if b == len(score.bars) - 1:
            out.append('      <barline location="right"><bar-style>light-heavy</bar-style></barline>')
        out.append('    </measure>')
    out += ['  </part>', '</score-partwise>']
    Path(path).write_text("\n".join(out), encoding="utf-8")
    return Path(path)

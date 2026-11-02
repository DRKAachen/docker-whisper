"""Prüft, dass api_server.py und diarizer.py zusammenpassen: Jeder Parameter, den der Server an diarizer.load() und
diarizer.diarize() übergibt, muss dort auch existieren (sonst startet die Sprecher-Erkennung nicht oder läuft ohne Fortschritt).
Läuft ohne sherpa-onnx (die Bibliothek wird für den Test durch eine Attrappe ersetzt).
Aufruf: python3 test_contract.py"""
import ast
import importlib.util
import inspect
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))


def find(name):
    return next(p for p in (os.path.join(HERE, "..", name), os.path.join(HERE, name)) if os.path.exists(p))


failed = 0


def check(name, cond, detail=""):
    global failed
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"   -> {detail}"))
    if not cond:
        failed += 1


sys.modules.setdefault("sherpa_onnx", types.ModuleType("sherpa_onnx"))
spec = importlib.util.spec_from_file_location("diarizer_contract", find("diarizer.py"))
dz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dz)

tree = ast.parse(open(find("api_server.py"), encoding="utf-8").read())
calls = {"load": [], "diarize": []}
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "diarizer":
        if node.func.attr in calls:
            calls[node.func.attr].append([k.arg for k in node.keywords if k.arg])
check("Der Server ruft diarizer.load() und diarizer.diarize() auf", bool(calls["load"]) and bool(calls["diarize"]), calls)
for fn in ("load", "diarize"):
    params = set(inspect.signature(getattr(dz, fn)).parameters)
    for kw in calls[fn]:
        missing = [k for k in kw if k not in params]
        check(f"diarizer.{fn}(): alle vom Server übergebenen Parameter {kw} gibt es", not missing, f"fehlt: {missing}")
check("diarize() hat den Parameter 'progress' (sonst gibt es keinen Fortschritt und keinen Abbruch)", "progress" in inspect.signature(dz.diarize).parameters, "")
check("Die Ausnahme heißt genau DiarizationAborted (der Server erkennt sie am Namen)", hasattr(dz, "DiarizationAborted") and dz.DiarizationAborted.__name__ == "DiarizationAborted" and issubclass(dz.DiarizationAborted, Exception), "")
src = open(find("api_server.py"), encoding="utf-8").read()
check("Der Server prüft den Namen der Ausnahme, die diarizer.py wirft", 'type(exc).__name__ == "DiarizationAborted"' in src, "")
for fn in ("resegment_by_word", "assign_speakers"):
    check(f"diarizer.{fn}() ist vorhanden", callable(getattr(dz, fn, None)), "")

print("\n" + (f"{failed} Prüfung(en) FEHLGESCHLAGEN" if failed else "Alle Prüfungen bestanden"))
sys.exit(1 if failed else 0)

"""保存済みの metrics.json（one_step / window / closed_loop / throughput）から、現在の GATES でゲートを判定し直して書き戻す。
使い方: .venv/bin/python tools/nss8k_rejudge.py <metrics.json>...   （モデルや評価の測り直しはしない）"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from nss8k import evaluate_lib as ev

for p in sys.argv[1:]:
    m = json.load(open(p))
    m["gates"] = ev.judge(m["kind"], m["one_step"], m["window"], m["closed_loop"], m["throughput"])
    m["gate_limits"] = ev.GATES
    json.dump(m, open(p, "w"), indent=1, ensure_ascii=False)
    print(p, "".join(k + ("○" if v["pass"] else "×") + " " for k, v in m["gates"].items()))

#!/bin/bash
cd "$(dirname "$0")/.."
.venv/bin/python python/gain_opt.py scan --runs analytic B_w64_s0 B_w64_s1 B_w64_s2 A_w64_s0 A_w64_s1 A_w64_s2 k0.9
echo SCANDONE

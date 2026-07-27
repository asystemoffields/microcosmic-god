#!/bin/bash
# Harvest the relaunched era-2.1 seed-45 pair (2026-07-27) as soon as each completes.
# The original arms' outputs were destroyed by the 06-13 no-op cancel stubs — this exists
# so the relaunch cannot be lost the same way. Cron */15; removes nothing, only downloads.
set -u
K=/data/kagglecli-venv/bin/kaggle
D=/data/microcosmic-god/kaggle
ST=$D/_harvest_e21
OWNER=asystemoffields
SLUGS="mg-e21-f50s45-c20 mg-e21-f10s45-c21"
mkdir -p "$ST"
[ -f "$ST/DONE" ] && exit 0

log() { echo "$(date -u +%F-%T) $*"; }
all_done=1
for s in $SLUGS; do
  [ -f "$ST/harvested_$s" ] && continue
  st=$("$K" kernels status "$OWNER/$s" 2>/dev/null | grep -oE 'KernelWorkerStatus\.[A-Z_]+' | head -1)
  case "$st" in
    *COMPLETE*)
      mkdir -p "$D/results/$s"
      if "$K" kernels output "$OWNER/$s" -p "$D/results/$s" --force --quiet; then
        # sanity: a real arm publishes run/ summaries, not just a log
        if compgen -G "$D/results/$s/run/*/summary.json" > /dev/null; then
          log "harvested $s"; date -u > "$ST/harvested_$s"
        else
          log "WARN $s output has no run/*/summary.json (stub or partial?)"; all_done=0
        fi
      else
        log "download failed for $s"; all_done=0
      fi ;;
    *ERROR*|*CANCEL*)
      log "TERMINAL-BAD $s: $st"; date -u > "$ST/FAILED_$s"; date -u > "$ST/harvested_$s" ;;
    *) log "$s ${st:-NO_STATUS}"; all_done=0 ;;
  esac
done
if [ "$all_done" = 1 ]; then
  ok=1
  for s in $SLUGS; do [ -f "$ST/harvested_$s" ] || ok=0; done
  [ "$ok" = 1 ] && { date -u > "$ST/DONE"; log "era-2.1 s45 pair harvest DONE"; }
fi
exit 0

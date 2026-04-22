#!/bin/bash
# Variance convergence sequence for H2O
# Runs a series of increasing subspace sizes, computing energy and variance at each step.
#
# Usage: bash variance_convergence.sh <diag_binary> <fcidump> <initial_adetfile>
#
# Output: prints a table of (step, n_dets, energy, variance, expanded_dets)

set -e

DIAG=${1:?Usage: $0 <diag_binary> <fcidump> <initial_adetfile>}
FCIDUMP=${2:?}
ADETFILE=${3:?}
WORKDIR=/tmp/sbd_variance_conv
NP=${NP:-1}  # MPI ranks, default 1
MPIRUN_EXTRA=${MPIRUN_EXTRA:-}  # e.g. "--allow-run-as-root"
THRESHOLD=${THRESHOLD:-0.001}   # amplitude threshold for S+D extend
CARRYOVER_TYPE=${CARRYOVER_TYPE:-4}  # 4=threshold S+D, 6=unfiltered S+D, 7=ERI-screened S+D, 8=ERI-screened all
ERI_THRESHOLD=${ERI_THRESHOLD:-1e-6}  # ERI screening threshold (types 7/8 only)
MAX_DETS=${MAX_DETS:-50000}     # stop if expanded dets exceed this
MAX_STEPS=${MAX_STEPS:-6}

SBD_LD_PATH="/opt/openmpi-5.0.8/lib:/opt/ohpc/pub/libs/gnu14/openblas/0.3.29/lib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$SBD_LD_PATH"

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

# Copy initial det file
cp "$ADETFILE" "$WORKDIR/step0_alpha.txt"

echo "================================================================"
echo " Variance Convergence Sequence"
echo " FCIDUMP: $FCIDUMP"
echo " Initial dets: $ADETFILE"
echo " Carryover type: $CARRYOVER_TYPE, threshold: $THRESHOLD, eri_threshold: $ERI_THRESHOLD"
echo " MPI ranks: $NP, max dets: $MAX_DETS"
echo "================================================================"
echo ""
printf "%-6s %10s %22s %16s %12s %10s\n" "Step" "N_dets" "Energy" "Variance" "Expanded" "Time(s)"
printf "%-6s %10s %22s %16s %12s %10s\n" "----" "------" "------" "--------" "--------" "------"

for step in $(seq 0 $MAX_STEPS); do
    STEP_ADET="$WORKDIR/step${step}_alpha.txt"

    if [ ! -f "$STEP_ADET" ]; then
        echo "No det file for step $step, stopping."
        break
    fi

    N_DETS=$(wc -l < "$STEP_ADET" | tr -d ' ')

    STEP_START=$(date +%s)

    # Run 1: Diagonalize + S+D expand
    STEP_WF="$WORKDIR/step${step}_wf_"
    STEP_EXPANDED="$WORKDIR/step$((step+1))_alpha.txt"

    CARRYOVER_ARGS="--carryover_type $CARRYOVER_TYPE"
    if [ "$CARRYOVER_TYPE" = "4" ] || [ "$CARRYOVER_TYPE" = "7" ]; then
        CARRYOVER_ARGS="$CARRYOVER_ARGS --carryover_threshold $THRESHOLD"
    fi
    if [ "$CARRYOVER_TYPE" = "7" ] || [ "$CARRYOVER_TYPE" = "8" ]; then
        CARRYOVER_ARGS="$CARRYOVER_ARGS --eri_threshold $ERI_THRESHOLD"
    fi

    # Load previous wavefunction as initial guess (steps > 0)
    LOAD_ARGS=""
    PREV_STEP=$((step - 1))
    PREV_WF="$WORKDIR/step${PREV_STEP}_wf_"
    if [ "$step" -gt 0 ] && [ -f "${PREV_WF}000000" ]; then
        LOAD_ARGS="--loadname $PREV_WF"
    fi

    mpirun $MPIRUN_EXTRA -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -np $NP "$DIAG" \
        --fcidump "$FCIDUMP" \
        --adetfile "$STEP_ADET" \
        --iteration 300 \
        $LOAD_ARGS \
        $CARRYOVER_ARGS \
        --savename "$STEP_WF" \
        --carryover_adetfile "$STEP_EXPANDED" \
        > "$WORKDIR/step${step}_diag.log" 2>&1

    DIAG_ENERGY=$(grep "^ Energy = " "$WORKDIR/step${step}_diag.log" | head -1 | awk '{print $NF}')

    # Count expanded dets (if generated)
    if [ -f "$STEP_EXPANDED" ]; then
        N_EXPANDED=$(wc -l < "$STEP_EXPANDED" | tr -d ' ')
    else
        N_EXPANDED="-"
    fi

    # Run 2: Variance-only on expanded space
    if [ -f "$STEP_EXPANDED" ] && [ "$N_EXPANDED" != "-" ] && [ "$N_EXPANDED" -le "$MAX_DETS" ]; then
        mpirun $MPIRUN_EXTRA -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -np $NP "$DIAG" \
            --fcidump "$FCIDUMP" \
            --adetfile "$STEP_EXPANDED" \
            --loadname "$STEP_WF" \
            --iteration 0 \
            > "$WORKDIR/step${step}_variance.log" 2>&1

        VARIANCE=$(grep "Energy variance" "$WORKDIR/step${step}_variance.log" | awk '{print $NF}')
    else
        VARIANCE="(skipped)"
    fi

    STEP_END=$(date +%s)
    ELAPSED=$((STEP_END - STEP_START))

    printf "%-6s %10s %22s %16s %12s %10s\n" "$step" "$N_DETS" "$DIAG_ENERGY" "$VARIANCE" "$N_EXPANDED" "$ELAPSED"

    # Stop if expanded exceeds limit (next diag would be too slow)
    if [ "$N_EXPANDED" != "-" ] && [ "$N_EXPANDED" -gt "$MAX_DETS" ]; then
        echo "Expanded dets ($N_EXPANDED) exceed limit ($MAX_DETS), stopping."
        break
    fi
done

echo ""
echo "Logs saved in $WORKDIR"
echo "================================================================"

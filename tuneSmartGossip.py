#!/usr/bin/env python3
"""
Parameter optimization for SMART_GOSSIP using differential evolution.

The formula is:
    p = max(p_floor, rssi_factor * density_factor * dupe_factor)

    rssi_factor    = 1 / (1 + exp(alpha * (rssi - rssi_ref)))
    density_factor = 1 / (1 + beta * n_neighbours)
    dupe_factor    = exp(-gamma * (dupes - 1))

Parameters optimized: [alpha, rssi_ref, beta, gamma, p_floor]

Objective: maximize composite_score = reachability - w_col*collision - w_tx*tx_air_util
           subject to: reachability >= REACH_MIN_THRESHOLD

Run with:  python3 tuneSmartGossip.py
"""

import random
import numpy as np
import simpy
from scipy.optimize import differential_evolution

from lib.config import Config
from lib.common import find_random_position, setup_asymmetric_links
from lib.discrete_event import BroadcastPipe
from lib.node import MeshNode, NodeConfig
from lib.point import Point

# ──────────────────────────────────────────────
# Optimization settings
# ──────────────────────────────────────────────
REPETITIONS = 5          # simulations per parameter set (increase for accuracy)
NR_NODES_EVAL = [15, 30] # which node counts to evaluate on

# Objective is a composite score:
#   score = W_REACH*reachability + W_USEFUL*usefulness - W_COL*collision - W_TX*tx_air_norm
# Weights can be tuned to emphasise different metrics.
W_REACH   = 0.40  # reward reachability
W_USEFUL  = 0.40  # reward packet usefulness (fraction of received pkts that were new)
W_COL     = 0.10  # penalise collision rate
W_TX      = 0.10  # penalise Tx air utilization (normalised to [0,1] by /100000)

# Differential evolution settings
POPSIZE   = 8
MAXITER   = 20
SEED_OPT  = 42

# Parameter bounds: [alpha, rssi_ref, beta, gamma, p_floor]
BOUNDS = [
    (0.02, 0.20),    # alpha:    RSSI sigmoid steepness
    (-120, -80),     # rssi_ref: inflection point (dBm)
    (0.01, 0.15),    # beta:     neighbour density weight
    (0.10, 0.60),    # gamma:    duplicate penalty
    (0.10, 0.40),    # p_floor:  minimum probability
]

# ──────────────────────────────────────────────
# Pre-generate node positions (same as batchSim)
# ──────────────────────────────────────────────
class TempNode:
    def __init__(self, conf, x, y):
        self.position = Point(x, y, conf.HM)


_base_conf = Config()
positions_cache = {}
for nrNodes in NR_NODES_EVAL:
    for rep in range(REPETITIONS):
        random.seed(rep)
        temp_nodes = []
        found = False
        while not found:
            temp_nodes = []
            for _ in range(nrNodes):
                xnew, ynew = find_random_position(_base_conf, temp_nodes)
                if xnew is None:
                    break
                temp_nodes.append(TempNode(_base_conf, xnew, ynew))
            if len(temp_nodes) == nrNodes:
                found = True
        positions_cache[(nrNodes, rep)] = [(t.position.x, t.position.y) for t in temp_nodes]


# ──────────────────────────────────────────────
# Single simulation run
# ──────────────────────────────────────────────
def run_simulation(nrNodes, rep, params):
    alpha, rssi_ref, beta, gamma, p_floor = params

    conf = Config()
    conf.SELECTED_ROUTER_TYPE = conf.ROUTER_TYPE.SMART_GOSSIP
    conf.NR_NODES = nrNodes
    conf.update_router_dependencies()
    conf.SEED = rep
    conf.SMART_GOSSIP_ALPHA    = alpha
    conf.SMART_GOSSIP_RSSI_REF = rssi_ref
    conf.SMART_GOSSIP_BETA     = beta
    conf.SMART_GOSSIP_GAMMA    = gamma
    conf.SMART_GOSSIP_P_FLOOR  = p_floor

    random.seed(rep)
    env     = simpy.Environment()
    bc_pipe = BroadcastPipe(env)

    coords  = positions_cache[(nrNodes, rep)]
    nodes, messages, packets, delays = [], [], [], []
    packetsAtN = [[] for _ in range(nrNodes)]
    messageSeq = {"val": 0}

    for nodeId in range(nrNodes):
        x, y = coords[nodeId]
        nc = NodeConfig(nodeId, Point(x, y, conf.HM), antenna_gain=conf.GL, hop_limit=conf.hopLimit)
        node = MeshNode(conf, nodes, env, bc_pipe, conf.PERIOD,
                        messages, packetsAtN, packets, delays, nc, messageSeq)
        nodes.append(node)

    setup_asymmetric_links(conf, nodes)
    env.run(until=conf.SIMTIME)

    nrCollisions = sum(1 for pkt in packets for n in nodes if pkt.collidedAtN[n.nodeid])
    nrSensed     = sum(1 for pkt in packets for n in nodes if pkt.sensedByN[n.nodeid])
    nrReceived   = sum(1 for pkt in packets for n in nodes if pkt.receivedAtN[n.nodeid])
    nrUseful     = sum(n.usefulPackets for n in nodes)

    collision_rate = float(nrCollisions) / nrSensed * 100 if nrSensed else float('nan')
    reachability   = nrUseful / (messageSeq["val"] * (nrNodes - 1)) * 100 if messageSeq["val"] else float('nan')
    usefulness     = float(nrUseful) / nrReceived * 100 if nrReceived else float('nan')
    tx_air_util    = sum(n.txAirUtilization for n in nodes) / nrNodes

    return reachability, collision_rate, usefulness, tx_air_util


# ──────────────────────────────────────────────
# Objective function (minimised by scipy)
# ──────────────────────────────────────────────
_eval_count = [0]

def objective(params):
    all_reach, all_col, all_useful, all_tx = [], [], [], []

    for nrNodes in NR_NODES_EVAL:
        for rep in range(REPETITIONS):
            reach, col, useful, tx = run_simulation(nrNodes, rep, params)
            if not np.isnan(reach):
                all_reach.append(reach)
                all_col.append(col)
                all_useful.append(useful)
                all_tx.append(tx)

    if not all_reach:
        return 1e9

    mean_reach  = np.mean(all_reach)
    mean_col    = np.mean(all_col)
    mean_useful = np.mean(all_useful)
    mean_tx     = np.mean(all_tx)

    # Composite score (all terms on comparable scale %)
    # tx is in ms — normalise to % scale by dividing by 100000 (100s = rough max per node)
    tx_pct = mean_tx / 1000.0  # ms → rough % scale
    score = (W_REACH * mean_reach
             + W_USEFUL * mean_useful
             - W_COL * mean_col
             - W_TX * tx_pct)

    _eval_count[0] += 1
    alpha, rssi_ref, beta, gamma, p_floor = params
    print(f"  eval {_eval_count[0]:4d} | reach={mean_reach:.1f}% useful={mean_useful:.1f}% "
          f"col={mean_col:.1f}% tx={mean_tx:.0f}ms | score={score:.2f} "
          f"| α={alpha:.3f} rssi_ref={rssi_ref:.1f} β={beta:.3f} γ={gamma:.3f} floor={p_floor:.3f}")

    return -score  # scipy minimises, we want to maximise


# ──────────────────────────────────────────────
# Run optimisation
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("SMART_GOSSIP parameter optimisation via differential evolution")
    print(f"  Node counts : {NR_NODES_EVAL}")
    print(f"  Repetitions : {REPETITIONS} per config")
    print(f"  Population  : {POPSIZE}  |  Max iterations: {MAXITER}")
    print("=" * 70)

    result = differential_evolution(
        objective,
        BOUNDS,
        seed=SEED_OPT,
        popsize=POPSIZE,
        maxiter=MAXITER,
        tol=0.01,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False,
        disp=True,
        workers=1,
    )

    alpha, rssi_ref, beta, gamma, p_floor = result.x
    print("\n" + "=" * 70)
    print("BEST PARAMETERS FOUND:")
    print(f"  alpha    = {alpha:.4f}    # RSSI sigmoid steepness")
    print(f"  rssi_ref = {rssi_ref:.2f}  # RSSI inflection point (dBm)")
    print(f"  beta     = {beta:.4f}    # neighbour density weight")
    print(f"  gamma    = {gamma:.4f}    # duplicate penalty")
    print(f"  p_floor  = {p_floor:.4f}    # minimum probability")
    print(f"\nBest composite score: {-result.fun:.4f}")
    print("=" * 70)
    print("\nCopy these values into lib/config.py as SMART_GOSSIP_* defaults.")

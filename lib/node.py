#!/usr/bin/env python3
from enum import Enum
import logging
import math
import random

import simpy

from lib.common import find_random_position
from lib.mac import set_transmit_delay, get_retransmission_msec
from lib.phy import check_collision, is_channel_active, airtime
from lib.packet import NODENUM_BROADCAST, MeshPacket, MeshMessage
from lib.point import Point

logger = logging.getLogger(__name__)

def generate_node_list(conf, node_configs, env, bc_pipe, period, messages, packetsAtN, packets, delays, messageSeq):
    """default function for randomly choosing node configurations for a simulation
    run, based on the provided config and desired number of nodes.

    We have lots of extra parameters that are only really necessary for MeshNode
    constructor, for tying it in to simulation state stuff. Needs refactoring
    """
    # need to identically match RNG usage right now to pass the discrete sim
    # test. If we want to change the reference test, do that in a smaller change.

    nodes = []

    # replicate default 'no prior config' setup:
    i = 0
    for n in node_configs:
        if n is None:
            # no specified node config, randomly generate one
            # get node's position
            x, y = find_random_position(conf, nodes)
            z = conf.HM
            position = Point(x, y, z)

            # role
            isRouter = conf.router
            isRepeater = False
            isClientMute = False

            # other default values
            hopLimit = conf.hopLimit
            antennaGain = conf.GL

            # map misc. booleans into single role
            if isRouter:
                role = MESHTASTIC_ROLE.ROUTER
            else:
                role = MESHTASTIC_ROLE.CLIENT

            # make NodeConfig object to pass to MeshNode constructor
            node_config = NodeConfig(i, position, role)
            i += 1

            # have node config, need to create node, add to list of nodes
            node = MeshNode(conf, nodes, env, bc_pipe, period, messages, packetsAtN, packets, delays, node_config, messageSeq)
            nodes.append(node)
        else:
            # convert dict from interactive GUI gen_scenario to NodeConfig objects, create nodes.
            node_config_dict = node_configs[n]
            position = Point(node_config_dict['x'], node_config_dict['y'], node_config_dict['z'])

            # roles
            isRouter = node_config_dict['isRouter']
            isRepeater = node_config_dict['isRepeater']
            isClientMute = node_config_dict['isClientMute']

            # sanity check that only one role is set
            if (isRouter and isRepeater) or \
               (isRepeater and isClientMute) or \
               (isClientMute and isRouter):
               raise Exception(f"invalid combination of roles: {node_config_dict}")

            if isRouter:
                role = MESHTASTIC_ROLE.ROUTER
            elif isRepeater:
                role = MESHTASTIC_ROLE.REPEATER
            elif isClientMute:
                role = MESHTASTIC_ROLE.CLIENT_MUTE
            else:
                role = MESHTASTIC_ROLE.CLIENT

            # make node config
            node_config = NodeConfig(i, position, role, node_config_dict['antennaGain'], node_config_dict['hopLimit'], node_config_dict['neighborInfo'])
            i += 1

            node = MeshNode(conf, nodes, env, bc_pipe, period, messages, packetsAtN, packets, delays, node_config, messageSeq)
            nodes.append(node)

    return nodes

# roles taken from the protobuf config meshtastic/config.proto in https://github.com/meshtastic/protobufs
# deprecated roles are included for simulation utility
class MESHTASTIC_ROLE(Enum):
    CLIENT = 'CLIENT'
    CLIENT_MUTE = 'CLIENT_MUTE'
    ROUTER = 'ROUTER'
    ROUTER_CLIENT = 'ROUTER_CLIENT'
    REPEATER = 'REPEATER'
    TRACKER = 'TRACKER'
    SENSOR = 'SENSOR'
    TAK = 'TAK'
    CLIENT_HIDDEN = 'CLIENT_HIDDEN'
    LOST_AND_FOUND = 'LOST_AND_FOUND'
    TAK_TRACKER = 'TAK_TRACKER'
    ROUTER_LATE = 'ROUTER_LATE'
    CLIENT_BASE = 'CLIENT_BASE'

class NodeConfig:
    """Specific configuration for a node
    """
    def __init__(self, node_id: int, position: Point, role: MESHTASTIC_ROLE = MESHTASTIC_ROLE.CLIENT, antenna_gain: float = 0, hop_limit: int = 3, neighbor_info: bool = False):
        self.node_id = node_id
        self.position = position.copy() # make sure we keep our own point
        self.role = role
        self.antenna_gain = antenna_gain
        self.hop_limit = hop_limit
        self.neighbor_info = neighbor_info

class MeshNode:
    """Class containing all the particular state of a MeshNode, references to necessary
    external resources like the simpy env, and process functions for simulation
    """
    def __init__(self, conf, nodes, env, bc_pipe, period, messages, packetsAtN, packets, delays, nodeConfig: NodeConfig, messageSeq):
        self.conf = conf
        self.nodeid = nodeConfig.node_id
        self.moveRng = random.Random(self.nodeid)
        self.nodeRng = random.Random(self.nodeid)
        self.rebroadcastRng = random.Random()
        self.neighbors = set()

        # require the user to specify a node configuration now, including position
        self.position = nodeConfig.position.copy() # make sure we have our own point
        self.role = nodeConfig.role
        self.hopLimit = nodeConfig.hop_limit
        self.antennaGain = nodeConfig.antenna_gain

        self.messageSeq = messageSeq
        self.env = env
        self.period = period
        self.bc_pipe = bc_pipe
        self.nodes = nodes
        self.messages = messages
        self.packetsAtN = packetsAtN
        self.nrPacketsSent = 0
        self.packets = packets
        self.delays = delays
        self.timesReceived = {}
        self.isReceiving = []
        self.isTransmitting = False
        self.usefulPackets = 0
        self.txAirUtilization = 0
        self.airUtilization = 0
        self.droppedByDelay = 0
        self.rebroadcastPackets = 0
        self.isMoving = False
        self.gpsEnabled = False
        # Track last broadcast position/time
        self.lastBroadcastPosition = self.position.copy()
        self.lastBroadcastTime = 0
        # track total transmit time for the last 6 buckets (each is 10s in firmware logic)
        self.channelUtilization = [0] * self.conf.CHANNEL_UTILIZATION_PERIODS  # each entry is ms spent on air in that interval
        self.channelUtilizationIndex = 0  # which "bucket" is current
        self.prevTxAirUtilization = 0.0   # how much total tx air-time had been used at last sample

        env.process(self.track_channel_utilization(env))
        if not self.is_repeater:  # repeaters don't generate messages themselves
            env.process(self.generate_message())
        env.process(self.receive(self.bc_pipe.get_output_conn()))
        self.transmitter = simpy.Resource(env, 1)

        # start mobility if enabled
        if self.conf.MOVEMENT_ENABLED and self.moveRng.random() <= self.conf.APPROX_RATIO_NODES_MOVING:
            self.isMoving = True
            if self.moveRng.random() <= self.conf.APPROX_RATIO_OF_NODES_MOVING_W_GPS_ENABLED:
                self.gpsEnabled = True

            # Randomly assign a movement speed
            possibleSpeeds = [
                self.conf.WALKING_METERS_PER_MIN,  # e.g.,  96 m/min
                self.conf.BIKING_METERS_PER_MIN,   # e.g., 390 m/min
                self.conf.DRIVING_METERS_PER_MIN   # e.g., 1500 m/min
            ]
            self.movementStepSize = self.moveRng.choice(possibleSpeeds)

            env.process(self.move_node(env))

    @property
    def is_router(self):
        return self.role == MESHTASTIC_ROLE.ROUTER

    @property
    def is_repeater(self):
        return self.role == MESHTASTIC_ROLE.REPEATER

    @property
    def is_client_mute(self):
        return self.role == MESHTASTIC_ROLE.CLIENT_MUTE

    def track_channel_utilization(self, env):
        """
        Periodically compute how many seconds of airtime this node consumed
        over the last 10-second block and store it in the ring buffer.
        """
        while True:
            # Wait 10 seconds of simulated time
            yield env.timeout(self.conf.TEN_SECONDS_INTERVAL)

            curTotalAirtime = self.txAirUtilization  # total so far, in *milliseconds*
            blockAirtimeMs = curTotalAirtime - self.prevTxAirUtilization

            self.channelUtilization[self.channelUtilizationIndex] = blockAirtimeMs

            self.prevTxAirUtilization = curTotalAirtime
            self.channelUtilizationIndex = (self.channelUtilizationIndex + 1) % self.conf.CHANNEL_UTILIZATION_PERIODS

    def channel_utilization_percent(self) -> float:
        """
        Returns how much of the last 60 seconds (6 x 10s) this node spent transmitting, as a percent.
        """
        sumMs = sum(self.channelUtilization)
        # 6 intervals, each 10 seconds = 60,000 ms total
        # fraction = sum_ms / 60000, then multiply by 100 for percent
        return (sumMs / (self.conf.CHANNEL_UTILIZATION_PERIODS * self.conf.TEN_SECONDS_INTERVAL)) * 100.0

    def move_node(self, env):
        while True:

            # Pick a random direction and distance
            angle = 2 * math.pi * self.moveRng.random()
            distance = self.movementStepSize * self.moveRng.random()

            # Compute new position
            dx = distance * math.cos(angle)
            dy = distance * math.sin(angle)

            leftBound = self.conf.OX - self.conf.XSIZE / 2
            rightBound = self.conf.OX + self.conf.XSIZE / 2
            bottomBound = self.conf.OY - self.conf.YSIZE / 2
            topBound = self.conf.OY + self.conf.YSIZE / 2

            # Then in moveNode:
            new_x = min(max(self.position.x + dx, leftBound), rightBound)
            new_y = min(max(self.position.y + dy, bottomBound), topBound)

            # Update node’s position
            self.position.update_xy(new_x, new_y)

            if self.gpsEnabled:
                distanceTraveled = self.position.euclidean_distance(self.lastBroadcastPosition)
                logger.debug(f"{self.env.now:.3f} node {self.nodeid} checks last broadcast position distance: {distanceTraveled} from {self.lastBroadcastPosition} to {self.position}")
                timeElapsed = env.now - self.lastBroadcastTime
                if distanceTraveled >= self.conf.SMART_POSITION_DISTANCE_THRESHOLD and timeElapsed >= self.conf.SMART_POSITION_DISTANCE_MIN_TIME:
                    currentUtil = self.channel_utilization_percent()
                    if currentUtil < 25.0:
                        self.send_packet(NODENUM_BROADCAST, "POSITION")
                        self.lastBroadcastPosition.update_xy(self.position.x, self.position.y)
                        self.lastBroadcastTime = env.now
                    else:
                        logger.debug(f"{self.env.now:.3f} node {self.nodeid} SKIPS POSITION broadcast (util={currentUtil:.1f}% > 25%)")

            # Wait until next move
            nextMove = self.get_next_time(self.conf.ONE_MIN_INTERVAL)
            if nextMove >= 0:
                yield env.timeout(nextMove)
            else:
                break

    def send_packet(self, destId, type=""):
        # increment the shared counter
        self.messageSeq["val"] += 1
        messageSeq = self.messageSeq["val"]
        self.messages.append(MeshMessage(self.nodeid, destId, self.env.now, messageSeq))
        p = MeshPacket(self.conf, self.nodes, self.nodeid, destId, self.nodeid, self.conf.PACKETLENGTH, messageSeq, self.env.now, True, False, None, self.env.now)
        logger.debug(f"{self.env.now:.3f} Node {self.nodeid} generated {type} message {p.seq} to {destId}")
        self.packets.append(p)
        self.env.process(self.transmit(p))
        return p

    def get_next_time(self, period):
        nextGen = self.nodeRng.expovariate(1.0 / float(period))
        # do not generate message near the end of the simulation (otherwise flooding cannot finish in time)
        if self.env.now+nextGen + self.hopLimit * airtime(self.conf, self.conf.current_preset["sf"], self.conf.current_preset["cr"], self.conf.PACKETLENGTH, self.conf.current_preset["bw"]) < self.conf.SIMTIME:
            return nextGen
        return -1
    

    def was_seen_recently(self, packet, ownTransmit=False):
        if packet.seq not in self.timesReceived:
            # First time we know about this packet
            self.timesReceived[packet.seq] = 0 if ownTransmit else 1
            if not ownTransmit:
                self.usefulPackets += 1
        else:
            self.timesReceived[packet.seq] += 0 if ownTransmit else 1


    def perhaps_cancel_dupe(self, packet):
        # Cancel if we've already seen this sequence number
        if packet.seq in self.timesReceived:
            return self.timesReceived[packet.seq] > 2 if self.is_router or self.is_repeater else self.timesReceived[packet.seq] > 1
        return False


    def generate_message(self):
        while True:
            # Returns -1 if we don't make it before the sim ends
            nextGen = self.get_next_time(self.period)
            # do not generate a message near the end of the simulation (otherwise flooding cannot finish in time)
            if nextGen >= 0:
                yield self.env.timeout(nextGen)

                if self.conf.DMs:
                    destId = self.nodeRng.choice([i for i in range(0, len(self.nodes)) if i is not self.nodeid])
                else:
                    destId = NODENUM_BROADCAST

                p = self.send_packet(destId)

                while p.wantAck:  # ReliableRouter: retransmit message if no ACK received after timeout
                    retransmissionMsec = get_retransmission_msec(self, p)
                    yield self.env.timeout(retransmissionMsec)

                    ackReceived = False  # check whether you received an ACK on the transmitted message
                    minRetransmissions = self.conf.maxRetransmission
                    for packetSent in self.packets:
                        if packetSent.origTxNodeId == self.nodeid and packetSent.seq == p.seq:
                            if packetSent.retransmissions < minRetransmissions:
                                minRetransmissions = packetSent.retransmissions
                            if packetSent.ackReceived:
                                ackReceived = True
                    if ackReceived:
                        logger.debug(f"{self.env.now:.3f} Node {self.nodeid} received ACK on generated message with seq. nr. {p.seq}")
                        break
                    else:
                        if minRetransmissions > 0:  # generate new packet with same sequence number
                            pNew = MeshPacket(self.conf, self.nodes, self.nodeid, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None, self.env.now)
                            pNew.retransmissions = minRetransmissions - 1
                            logger.debug(f"{self.env.now:.3f} Node {self.nodeid} wants to retransmit its generated packet to {destId} with seq.nr. {p.seq} minRetransmissions {minRetransmissions}")
                            self.packets.append(pNew)
                            self.env.process(self.transmit(pNew))
                        else:
                            logger.debug(f"{self.env.now:.3f} Node {self.nodeid} reliable send of {p.seq} failed.")
                            break
            else:  # do not send this message anymore, since it is close to the end of the simulation
                break

    def transmit(self, packet):
        with self.transmitter.request() as request:
            yield request

            # listen-before-talk from src/mesh/RadioLibInterface.cpp
            txTime = set_transmit_delay(self, packet)
            logger.debug(f"{self.env.now:.3f} Node {self.nodeid} picked wait time {txTime}")
            yield self.env.timeout(txTime)

            # wait when currently receiving or transmitting, or channel is active
            while any(self.isReceiving) or self.isTransmitting or is_channel_active(self, self.env):
                logger.debug(f"{self.env.now:.3f} Node {self.nodeid} is busy Tx-ing {self.isTransmitting} or Rx-ing {any(self.isReceiving)} else channel busy!")
                txTime = set_transmit_delay(self, packet)
                yield self.env.timeout(txTime)
            logger.debug(f"{self.env.now:.3f} Node {self.nodeid} ends waiting")

            # check if you received an ACK for this message in the meantime
            self.was_seen_recently(packet, ownTransmit=True)
            if not self.perhaps_cancel_dupe(packet):  # if you did not receive an ACK for this message in the meantime
                logger.debug(f"{self.env.now:.3f} Node {self.nodeid} started low level send {packet.seq} hopLimit {packet.hopLimit} original Tx {packet.origTxNodeId}")
                self.nrPacketsSent += 1
                for rx_node in self.nodes:
                    if packet.sensedByN[rx_node.nodeid]:
                        if check_collision(self.conf, self.env, packet, rx_node.nodeid, self.packetsAtN) == 0:
                            self.packetsAtN[rx_node.nodeid].append(packet)
                packet.startTime = self.env.now
                packet.endTime = self.env.now + packet.timeOnAir
                self.txAirUtilization += packet.timeOnAir
                self.airUtilization += packet.timeOnAir
                self.bc_pipe.put(packet)
                self.isTransmitting = True
                yield self.env.timeout(packet.timeOnAir)
                self.isTransmitting = False
            else:  # received ACK: abort transmit, remove from packets generated
                logger.debug(f"{self.env.now:.3f} Node {self.nodeid} in the meantime received ACK, abort packet with seq. nr {packet.seq}")
                self.packets.remove(packet)

    def do_rebroadcast(self, packet):
        pNew = MeshPacket(self.conf, self.nodes, packet.origTxNodeId, packet.destId, self.nodeid, packet.packetLen, packet.seq, packet.genTime, packet.wantAck, False, None, self.env.now)
        pNew.hopLimit = packet.hopLimit - 1
        self.packets.append(pNew)
        self.env.process(self.transmit(pNew))

    def receive(self, in_pipe):
        while True:
            p = yield in_pipe.get()
            if p.sensedByN[self.nodeid] and not p.collidedAtN[self.nodeid] and p.onAirToN[self.nodeid]:  # start of reception
                if not self.isTransmitting:
                    logger.debug(f"{self.env.now:.3f} Node {self.nodeid} started receiving packet {p.seq} from {p.txNodeId}")
                    p.onAirToN[self.nodeid] = False
                    self.isReceiving.append(True)
                else:  # if you were currently transmitting, you could not have sensed it
                    logger.debug(f"{self.env.now:.3f} Node {self.nodeid} was transmitting, so could not receive packet {p.seq}")
                    p.sensedByN[self.nodeid] = False
                    p.onAirToN[self.nodeid] = False
            elif p.sensedByN[self.nodeid]:  # end of reception
                try:
                    self.isReceiving[self.isReceiving.index(True)] = False
                except Exception:
                    pass
                self.airUtilization += p.timeOnAir
                if p.collidedAtN[self.nodeid]:
                    logger.debug(f"{self.env.now:.3f} Node {self.nodeid} could not decode packet.")
                    continue
                p.receivedAtN[self.nodeid] = True
                logger.debug(f"{self.env.now:.3f} Node {self.nodeid} received packet {p.seq} with delay {round(self.env.now - p.genTime, 2)}") # TODO: better way to calculate delay for log
                self.delays.append(self.env.now - p.genTime)

                # Update history of received packets
                self.was_seen_recently(p)

                # check if implicit ACK for own generated message
                if p.origTxNodeId == self.nodeid:
                    if p.isAck:
                        logger.debug(f"Node {self.nodeid} received real ACK on generated message.")
                    else:
                        logger.debug(f"Node {self.nodeid} received implicit ACK on message sent.")
                    p.ackReceived = True
                    continue

                ackReceived = False
                realAckReceived = False
                for sentPacket in self.packets:
                    # check if ACK for message you currently have in queue
                    if sentPacket.txNodeId == self.nodeid and sentPacket.seq == p.seq:
                        logger.debug(f"{self.env.now:.3f} Node {self.nodeid} received implicit ACK for message in queue.")
                        ackReceived = True
                        sentPacket.ackReceived = True
                    # check if real ACK for message sent
                    if sentPacket.origTxNodeId == self.nodeid and p.isAck and sentPacket.seq == p.requestId:
                        logger.debug(f"{self.env.now:.3f} Node {self.nodeid} received real ACK.")
                        realAckReceived = True
                        sentPacket.ackReceived = True

                # send real ACK if you are the destination and you did not yet send the ACK
                if p.wantAck and p.destId == self.nodeid and not any(pA.requestId == p.seq for pA in self.packets):
                    logger.debug(f"{self.env.now:.3f} Node {self.nodeid} sends a flooding ACK.")
                    self.messageSeq["val"] += 1
                    messageSeq = self.messageSeq["val"]
                    self.messages.append(MeshMessage(self.nodeid, p.origTxNodeId, self.env.now, messageSeq))
                    pAck = MeshPacket(self.conf, self.nodes, self.nodeid, p.origTxNodeId, self.nodeid, self.conf.ACKLENGTH, messageSeq, self.env.now, False, True, p.seq, self.env.now)
                    self.packets.append(pAck)
                    self.env.process(self.transmit(pAck))

                # Rebroadcasting Logic for received message. This is a broadcast or a DM not meant for us.
                elif not p.destId == self.nodeid and not ackReceived and not realAckReceived and p.hopLimit > 0:
                    # FloodingRouter: rebroadcast received packet
                    if self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.MANAGED_FLOOD:
                        if not self.is_client_mute:
                            logger.debug(f"{self.env.now:.3f} Node {self.nodeid} MANAGED FLOODING rebroadcasts received packet {p.seq}")
                            self.do_rebroadcast(p)


                    elif self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.GOSSIP02:
                        # rebroadcast with a certain probability
                        p_prob = 0.2
                        if not self.is_client_mute and random.random() < p_prob:
                            logger.debug(f"{self.env.now:.3f} Node {self.nodeid} GOSSIP rebroadcast {p.seq}")
                            self.do_rebroadcast(p)

                    elif self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.GOSSIP04:
                        # rebroadcast with a certain probability
                        p_prob = 0.4
                        if not self.is_client_mute and random.random() < p_prob:
                            logger.debug(f"{self.env.now:.3f} Node {self.nodeid} GOSSIP rebroadcast {p.seq}")
                            self.do_rebroadcast(p)

                    elif self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.GOSSIP06:
                        # rebroadcast with a certain probability
                        p_prob = 0.6
                        if not self.is_client_mute and random.random() < p_prob:
                            logger.debug(f"{self.env.now:.3f} Node {self.nodeid} GOSSIP rebroadcast {p.seq}")
                            self.do_rebroadcast(p)

                    elif self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.GOSSIP08:
                        # rebroadcast with a certain probability
                        p_prob = 0.8
                        if not self.is_client_mute and random.random() < p_prob:
                            logger.debug(f"{self.env.now:.3f} Node {self.nodeid} GOSSIP rebroadcast {p.seq}")
                            self.do_rebroadcast(p)

                    elif self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.MANAGED_GOSSIP:
                        if not self.is_client_mute:
                            if self.timesReceived[p.seq] == 0:
                                p_prob = 0.9
                            elif self.timesReceived[p.seq] < 3:
                                p_prob = 0.6
                            else:
                                p_prob = 0.3
                            
                            if random.random() < p_prob:
                                logger.debug(f"{self.env.now:.3f} Node {self.nodeid} MANAGED GOSSIP rebroadcast {p.seq} (count={self.timesReceived[p.seq]})")
                                self.do_rebroadcast(p)
    
                    elif self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.ADAPTIVE_GOSSIP:
                        # rebroadcast with a probability that decreases as you receive more copies of the same message (including the original transmission, if you are a router/repeater). This is similar to Managed Gossip, but applied to all nodes and with a different probability function.
                        if p.txNodeId not in self.neighbors:
                            self.neighbors.add(p.txNodeId)
                            # print(f"{self.env.now:.3f} Node {self.nodeid} adds {p.txNodeId} to neighbors (total neighbors: {len(self.neighbors)})")
                        if not self.is_client_mute:
                            if len(self.neighbors) < 5:
                                base_p = 1.0
                            elif len(self.neighbors) < 15:
                                base_p = 0.9
                            else:
                                base_p = 0.8

                            p_prob = max(0.2, base_p - 0.05 * self.timesReceived[p.seq])  # e.g., start at 1.0, decrease by 0.1 for each copy received, but never go below 0.1
                            if random.random() < p_prob:
                                logger.debug(f"{self.env.now:.3f} Node {self.nodeid} ADAPTIVE GOSSIP rebroadcast {p.seq} (count={self.timesReceived[p.seq]})")
                                self.do_rebroadcast(p)



                    elif self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.COUNTER_BASED:
                        # rebroadcast if you received less than C copies of this message so far (including the original transmission, if you are a router/repeater)
                        C : int = 2

                        if not self.is_client_mute and self.timesReceived[p.seq] < C:
                            logger.debug(f"{self.env.now:.3f} Node {self.nodeid} COUNTER rebroadcast {p.seq} (count={self.timesReceived[p.seq]})")
                            self.do_rebroadcast(p)

                    elif self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.RSSI_BASED:
                                            # Probabilistic rebroadcast: the worse the RSSI, the higher the
                                            # probability of rebroadcasting (weak signal => more help needed).
                                            rssi_min = -120  # RSSI value at/below which P(rebroadcast) = max_prob
                                            rssi_max = -60   # RSSI value at/above which P(rebroadcast) = min_prob
                                            min_prob = 0.05  # floor probability even with great signal
                                            max_prob = 1.0   # ceiling probability with very bad signal

                                            rssi = p.rssiAtN[self.nodeid]

                                            # Clamp rssi into [rssi_min, rssi_max] then normalize to [0,1]
                                            clamped = max(rssi_min, min(rssi, rssi_max))
                                            normalized = (clamped - rssi_min) / (rssi_max - rssi_min)

                                            # Invert so worse RSSI -> higher probability
                                            rebroadcast_prob = max_prob - normalized * (max_prob - min_prob)

                                            if not self.is_client_mute and random.random() < rebroadcast_prob:
                                                logger.debug(f"{self.env.now:.3f} Node {self.nodeid} RSSI rebroadcast {p.seq} (rssi={rssi}, p={rebroadcast_prob:.2f})")
                                                self.do_rebroadcast(p)


                    elif self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.FLOODING_WITH_MEMORY:
                        # rebroadcast if you have not already received this message more than 2 times (including the original transmission, if you are a router/repeater). This is similar to Counter-Based, but with a higher threshold and applied to clients as well, since they can also help with rebroadcasting.
                        if not self.is_client_mute and self.timesReceived[p.seq] == 1:
                            logger.debug(f"{self.env.now:.3f} Node {self.nodeid} MEMORY rebroadcast {p.seq}")
                            self.do_rebroadcast(p)

                    elif self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.SMART_GOSSIP:
                        # Smooth formula: p = max(p_floor, rssi_factor * density_factor * dupe_factor)
                        # rssi_factor   = sigmoid(alpha, rssi, rssi_ref)  — weak signal → high p (edge node needs relay)
                        # density_factor = 1/(1+beta*n)                   — many neighbours → redundant → lower p
                        # dupe_factor   = exp(-gamma*(dupes-1))            — repeated copies → already covered → lower p
                        self.neighbors.add(p.txNodeId)
                        if not self.is_client_mute:
                            rssi = p.rssiAtN[self.nodeid]
                            n = len(self.neighbors)
                            dupes = self.timesReceived[p.seq]

                            alpha   = self.conf.SMART_GOSSIP_ALPHA
                            rssi_ref = self.conf.SMART_GOSSIP_RSSI_REF
                            beta    = self.conf.SMART_GOSSIP_BETA
                            gamma   = self.conf.SMART_GOSSIP_GAMMA
                            p_floor = self.conf.SMART_GOSSIP_P_FLOOR

                            rssi_factor    = 1.0 / (1.0 + math.exp(alpha * (rssi - rssi_ref)))
                            density_factor = 1.0 / (1.0 + beta * n)
                            dupe_factor    = math.exp(-gamma * (dupes - 1))

                            p_prob = max(p_floor, rssi_factor * density_factor * dupe_factor)
                            if random.random() < p_prob:
                                logger.debug(f"{self.env.now:.3f} Node {self.nodeid} SMART GOSSIP rebroadcast {p.seq} (rssi={rssi:.1f}, n={n}, dupes={dupes}, p={p_prob:.3f})")
                                self.do_rebroadcast(p)
                else:
                    self.droppedByDelay += 1

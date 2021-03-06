from __future__ import print_function

# Our Q-learning is based off Microsoft's tabular Q-learning located at https://github.com/microsoft/malmo
# Our Agents goal here it to find the end goal, while avoiding enetering the vision radius of any monsters and minimizing the steps of the enemy
#


# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# The "Cliff Walking" example using Q-learning.
# From pages 148-150 of:
# Richard S. Sutton and Andrews G. Barto
# Reinforcement Learning, An Introduction
# MIT Press, 1998

from future import standard_library

standard_library.install_aliases()
from builtins import input
from builtins import range
from builtins import object
import MalmoPython
import json
import logging
import math
import os
import random
import sys
import time
import malmoutils

import matplotlib.pyplot as plt

if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk

save_images = False
if save_images:
    from PIL import Image

malmoutils.fix_print()


class TabQAgent(object):
    """Tabular Q-learning agent for discrete state/action spaces."""

    def __init__(self, actions=[], epsilon=0.1, alpha=0.1, gamma=1.0, debug=False, canvas=None, root=None):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.training = True

        self.logger = logging.getLogger(__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.actions = actions
        self.q_table = {}
        self.canvas = canvas
        self.root = root

        self.rep = 0

    def loadModel(self, model_file):
        """load q table from model_file"""
        with open(model_file) as f:
            self.q_table = json.load(f)

    def training(self):
        """switch to training mode"""
        self.training = True

    def evaluate(self):
        """switch to evaluation mode (no training)"""
        self.training = False

    def act(self, world_state, agent_host, current_r):
        """take 1 action in response to the current world state"""
        
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text)  # most recent observation
        print(obs)
        print(current_r)
        self.logger.debug(obs)
        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        current_s = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
        self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'ZPos'])))
        if current_s not in self.q_table:
            self.q_table[current_s] = ([0] * len(self.actions))

        # update Q values
        if self.training and self.prev_s is not None and self.prev_a is not None:
            old_q = self.q_table[self.prev_s][self.prev_a]
            self.q_table[self.prev_s][self.prev_a] = old_q + self.alpha * (current_r
                                                                           + self.gamma * max(
                        self.q_table[current_s]) - old_q)

        self.drawQ(curr_x=int(obs[u'XPos']), curr_y=int(obs[u'ZPos']))

        #Special: if current_r is -76 (wall) then we go backwards
        if current_r == -76:
            # send the selected action
            # actionSet = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
            if self.prev_a == 0:
                a = 1
            elif self.prev_a == 1:
                a = 0
            elif self.prev_a == 2:
                a = 3
            elif self.prev_a == 3:
                a = 2
            agent_host.sendCommand(self.actions[a])
            self.prev_s = current_s
            self.prev_a = a
            return 100

        # otherwise, do normal q learning
        # select the next action
        rnd = random.random()
        if rnd < self.epsilon:
            a = random.randint(0, len(self.actions) - 1)
            self.logger.info("Random action: %s" % self.actions[a])
        else:
            m = max(self.q_table[current_s])
            self.logger.debug("Current values: %s" % ",".join(str(x) for x in self.q_table[current_s]))
            l = list()
            for x in range(0, len(self.actions)):
                if self.q_table[current_s][x] == m:
                    l.append(x)
            y = random.randint(0, len(l) - 1)
            a = l[y]
            self.logger.info("Taking q action: %s" % self.actions[a])

        # send the selected action
        agent_host.sendCommand(self.actions[a])
        self.prev_s = current_s
        self.prev_a = a

        return current_r

    def run(self, agent_host):
        """run the agent on the world"""

        total_reward = 0
        current_r = 0
        tol = 0.01

        self.prev_s = None
        self.prev_a = None

        # wait for a valid observation
        world_state = agent_host.peekWorldState()
        while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
            world_state = agent_host.peekWorldState()
        # wait for a frame to arrive after that
        num_frames_seen = world_state.number_of_video_frames_since_last_state
        while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
            world_state = agent_host.peekWorldState()
        world_state = agent_host.getWorldState()
        for err in world_state.errors:
            print(err)

        if not world_state.is_mission_running:
            return 0  # mission already ended

        assert len(world_state.video_frames) > 0, 'No video frames!?'

        obs = json.loads(world_state.observations[-1].text)
        prev_x = obs[u'XPos']
        prev_z = obs[u'ZPos']
        print('Initial position:', prev_x, ',', prev_z)

        if save_images:
            # save the frame, for debugging
            frame = world_state.video_frames[-1]
            image = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels))
            iFrame = 0
            self.rep = self.rep + 1
            image.save('rep_' + str(self.rep).zfill(3) + '_saved_frame_' + str(iFrame).zfill(4) + '.png')

        # take first action
        total_reward += self.act(world_state, agent_host, current_r)

        require_move = True
        check_expected_position = True

        # main loop:
        while world_state.is_mission_running:

            # wait for the position to have changed and a reward received
            print('Waiting for data...', end=' ')
            while True:
                world_state = agent_host.peekWorldState()
                if not world_state.is_mission_running:
                    print('mission ended.')
                    break
                if len(world_state.rewards) > 0 and not all(e.text == '{}' for e in world_state.observations):
                    obs = json.loads(world_state.observations[-1].text)
                    curr_x = obs[u'XPos']
                    curr_z = obs[u'ZPos']
                    if require_move:
                        if math.hypot(curr_x - prev_x, curr_z - prev_z) > tol:
                            print('received.')
                            break
                    else:
                        print('received.')
                        break
            # wait for a frame to arrive after that
            num_frames_seen = world_state.number_of_video_frames_since_last_state
            while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
                world_state = agent_host.peekWorldState()

            num_frames_before_get = len(world_state.video_frames)

            world_state = agent_host.getWorldState()
            for err in world_state.errors:
                print(err)
            current_r = sum(r.getValue() for r in world_state.rewards)

            if save_images:
                # save the frame, for debugging
                if world_state.is_mission_running:
                    assert len(world_state.video_frames) > 0, 'No video frames!?'
                    frame = world_state.video_frames[-1]
                    image = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels))
                    iFrame = iFrame + 1
                    image.save('rep_' + str(self.rep).zfill(3) + '_saved_frame_' + str(iFrame).zfill(4) + '_after_' +
                               self.actions[self.prev_a] + '.png')

            if world_state.is_mission_running:
                assert len(world_state.video_frames) > 0, 'No video frames!?'
                num_frames_after_get = len(world_state.video_frames)
                assert num_frames_after_get >= num_frames_before_get, 'Fewer frames after getWorldState!?'
                frame = world_state.video_frames[-1]
                obs = json.loads(world_state.observations[-1].text)
                curr_x = obs[u'XPos']
                curr_z = obs[u'ZPos']
                print('New position from observation:', curr_x, ',', curr_z, 'after action:', self.actions[self.prev_a],
                      end=' ')  # NSWE
                if check_expected_position:
                    expected_x = prev_x + [0, 0, -1, 1][self.prev_a]
                    expected_z = prev_z + [-1, 1, 0, 0][self.prev_a]
                    if math.hypot(curr_x - expected_x, curr_z - expected_z) > tol:
                        print(' - ERROR DETECTED! Expected:', expected_x, ',', expected_z)
                        input("Press Enter to continue...")
                    else:
                        print('as expected.')
                    curr_x_from_render = frame.xPos
                    curr_z_from_render = frame.zPos
                    print('New position from render:', curr_x_from_render, ',', curr_z_from_render, 'after action:',
                          self.actions[self.prev_a], end=' ')  # NSWE
                    if math.hypot(curr_x_from_render - expected_x, curr_z_from_render - expected_z) > tol:
                        print(' - ERROR DETECTED! Expected:', expected_x, ',', expected_z)
                        input("Press Enter to continue...")
                    else:
                        print('as expected.')
                else:
                    print()
                #input("Press Enter to continue...")
                prev_x = curr_x
                prev_z = curr_z
                # act
                total_reward += self.act(world_state, agent_host, current_r)

        # process final reward
        self.logger.debug("Final reward: %d" % current_r)
        total_reward += current_r

        # update Q values
        if self.training and self.prev_s is not None and self.prev_a is not None:
            old_q = self.q_table[self.prev_s][self.prev_a]
            self.q_table[self.prev_s][self.prev_a] = old_q + self.alpha * (current_r - old_q)

        self.drawQ()

        return total_reward

    def drawQ(self, curr_x=None, curr_y=None):
        if self.canvas is None or self.root is None:
            return
        self.canvas.delete("all")
        action_inset = 0.1
        action_radius = 0.1
        curr_radius = 0.2
        action_positions = [(0.5, 1 - action_inset), (0.5, action_inset), (1 - action_inset, 0.5), (action_inset, 0.5)]
        # (NSWE to match action order)
        min_value = -20
        max_value = 20
        for x in range(world_x):
            for y in range(world_y):
                s = "%d:%d" % (x, y)
                self.canvas.create_rectangle((world_x - 1 - x) * scale, (world_y - 1 - y) * scale,
                                             (world_x - 1 - x + 1) * scale, (world_y - 1 - y + 1) * scale,
                                             outline="#fff", fill="#000")
                for action in range(4):
                    if not s in self.q_table:
                        continue
                    value = self.q_table[s][action]
                    color = int(255 * (value - min_value) / (max_value - min_value))  # map value to 0-255
                    color = max(min(color, 255), 0)  # ensure within [0,255]
                    color_string = '#%02x%02x%02x' % (255 - color, color, 0)
                    self.canvas.create_oval((world_x - 1 - x + action_positions[action][0] - action_radius) * scale,
                                            (world_y - 1 - y + action_positions[action][1] - action_radius) * scale,
                                            (world_x - 1 - x + action_positions[action][0] + action_radius) * scale,
                                            (world_y - 1 - y + action_positions[action][1] + action_radius) * scale,
                                            outline=color_string, fill=color_string)
        if curr_x is not None and curr_y is not None:
            self.canvas.create_oval((world_x - 1 - curr_x + 0.5 - curr_radius) * scale,
                                    (world_y - 1 - curr_y + 0.5 - curr_radius) * scale,
                                    (world_x - 1 - curr_x + 0.5 + curr_radius) * scale,
                                    (world_y - 1 - curr_y + 0.5 + curr_radius) * scale,
                                    outline="#fff", fill="#fff")
        self.root.update()

def add_enemies(arena_width,arena_height, no_items):
    xml = ""
    # add more enemies, but avoid end goal
    used_pos = set((arena_width, arena_height-3))
    
    smaller_dim = min(arena_width, arena_height)
    enemies = (smaller_dim-1)//3
    
    for i in range(enemies):
        # now avoid start goal
        #x = random.randint(2, arena_width)
        #z = random.randint(0, arena_height-2)
        #if z == 1 and x in range(3, 4+3):
            #x = random.randint(1, arena_width-2)
        while True:
            x = random.randint(2, arena_width)
            z = random.randint(0, arena_height-4)
            while (z <= 2 ) and x in range(3, 4+3):
                x = random.randint(2, arena_width)
                z = random.randint(0, arena_height-4)
            if (x,z) not in used_pos:
                break
        
        used_pos.add((x,z))
        no_items.add((x-1,z+1))
        xml += '''<DrawCuboid x1="''' + str(x) + '''" y1="45" z1="''' + str(z) + '''" x2="''' + str(x-2) + '''" y2="45" z2="''' + str(z+2) + '''" type="red_sandstone"/>'''
        xml += '''<DrawEntity x="''' + str(x-0.5) + '''" y="45" z="''' + str(z+1.5) + '''"  type="Villager" />'''
    return xml

def add_items(arena_width, arena_height, items, no_items):
    xml = ""
    for i in range(items):
        
        while True:
            x = random.randint(0, arena_width)
            z = random.randint(0, arena_height-1)
            if (x,z) not in no_items:
                break
            
        no_items.add((x,z))
        xml += '''<DrawItem x="''' + str(x) + '''" y="46" z="''' + str(z) + '''" type="diamond" />'''
    return xml
    
def XML_generator(x,y,items):
    arena_width=x-1
    arena_height=y
    no_items = set((arena_width, arena_height-1))
    no_items.add((4,1))
    xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                
                  <About>
                    <Summary>Avoiding enemies to get to target.</Summary>
                  </About>
                  
                  <ModSettings>
                    <MsPerTick>1</MsPerTick>
                  </ModSettings>
                
                  <ServerSection>
                      <ServerInitialConditions>
                            <Time>
                                <StartTime>6000</StartTime>
                                <AllowPassageOfTime>false</AllowPassageOfTime>
                            </Time>
                            <Weather>clear</Weather>
                            <AllowSpawning>false</AllowSpawning>
                      </ServerInitialConditions>
                    <ServerHandlers>
                      <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1"/>
                      <DrawingDecorator>
                        
                          <!-- coordinates for cuboid are inclusive -->
                          <DrawCuboid x1="-2" y1="46" z1="-2" x2="'''+str(arena_width+2)+'''" y2="50" z2="'''+str(arena_height+2)+'''" type="air" />            <!-- limits of our arena -->
                          <DrawCuboid x1="-2" y1="45" z1="-2" x2="'''+str(arena_width+2)+'''" y2="45" z2="'''+str(arena_height+2)+'''" type="lava" />           <!-- lava floor -->
                          <DrawCuboid x1="-1"  y1="44" z1="0"  x2="'''+str(arena_width)+'''" y2="45" z2="'''+str(arena_height)+'''" type="sandstone" />      <!-- floor of the arena -->
                		  
                          <DrawBlock  x="4"   y="45"  z="1"  type="cobblestone" />                           <!-- the starting marker -->
                    		  
                          <!-- Boundary -->
                          <DrawCuboid x1="'''+str(arena_width+1)+'''"  y1="45" z1="-1"  x2="'''+str(arena_width+1)+'''" y2="45" z2="'''+str(arena_height)+'''" type="gold_block" />           <!-- Left wall from start position -->
                          <DrawCuboid x1="-1"  y1="45" z1="-1"  x2="'''+str(arena_width+1)+'''" y2="45" z2="-1" type="gold_block" />			  <!-- Bottom wall from start position -->
                          <DrawCuboid x1="-1"  y1="45" z1="-1"  x2="-1" y2="45" z2="'''+str(arena_height)+'''" type="gold_block" />           <!-- Right wall from start position -->
                          <DrawCuboid x1="-1"  y1="45" z1="'''+str(arena_height)+'''"  x2="'''+str(arena_width+1)+'''" y2="45" z2="'''+str(arena_height)+'''" type="gold_block" />           <!-- Top wall from start position -->
                
                          <!-- Goal -->
                          <DrawBlock  x="''' + str(arena_width) + '''"   y="45"  z="''' + str(arena_height-1) + '''" type="lapis_block" />
                		  
                          <!-- Enemies -->
                          '''+ add_enemies(arena_width,arena_height, no_items) + '''
                		  
                      </DrawingDecorator>
                      <ServerQuitFromTimeUp timeLimitMs="2000000"/>
                      <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                  </ServerSection>
                
                  <AgentSection mode="Survival">
                    <Name>Master</Name>
                    <AgentStart>
                      <Placement x="4.5" y="46.0" z="1.5" pitch="30" yaw="0"/>
                      <Inventory>
                      </Inventory>
                    </AgentStart>
                    <AgentHandlers>
                      <ObservationFromFullStats/>
                      <VideoProducer want_depth="false">
                          <Width>640</Width>
                          <Height>480</Height>
                      </VideoProducer>
                      <DiscreteMovementCommands>
                          <ModifierList type="deny-list">
                            <command>attack</command>
                          </ModifierList>
                      </DiscreteMovementCommands>
                      <RewardForTouchingBlockType>
                        <Block reward="-1000.0" type="lava" behaviour="onceOnly"/>
                        <Block reward="1000.0" type="lapis_block" behaviour="onceOnly"/>
                        <Block reward="-1000.0" type="red_sandstone" behaviour="onceOnly"/>
                        <Block reward="-75.0" type="gold_block"/>
                      </RewardForTouchingBlockType>
                      <RewardForSendingCommand reward="-1"/>
                      <RewardForCollectingItem>
                        <Item reward="100.0" type="diamond" />
                      </RewardForCollectingItem>
                      <AgentQuitFromTouchingBlockType>
                          <Block type="lava" />
                          <Block type="lapis_block" />
                          <Block type="red_sandstone" />
                		  <Block type="stone" />
                      </AgentQuitFromTouchingBlockType>
                    </AgentHandlers>
                  </AgentSection>
                
                </Mission>'''
    return xml



agent_host = MalmoPython.AgentHost()

# Find the default mission file by looking next to the schemas folder:
schema_dir = None
try:
    schema_dir = os.environ['MALMO_XSD_PATH']
except KeyError:
    print("MALMO_XSD_PATH not set? Check environment.")
    exit(1)
    
mission_file = os.path.abspath(os.path.join(schema_dir, '..',
                                            'sample_missions', 'cliff_walking_1.xml'))  # Integration test path

if not os.path.exists(mission_file):
    mission_file = os.path.abspath(os.path.join(schema_dir, '..',
                                                'Sample_missions', 'cliff_walking_1.xml'))  # Install path

if not os.path.exists(mission_file):
    print("Could not find cliff_walking_1.xml under MALMO_XSD_PATH")
    exit(1)

# add some args
agent_host.addOptionalStringArgument('mission_file',
                                     'Path/to/file from which to load the mission.', mission_file)
agent_host.addOptionalFloatArgument('alpha',
                                    'Learning rate of the Q-learning agent.', 0.1)
agent_host.addOptionalFloatArgument('epsilon',
                                    'Exploration rate of the Q-learning agent.', 0.01)
agent_host.addOptionalFloatArgument('gamma', 'Discount factor.', 1.0)
agent_host.addOptionalFlag('load_model', 'Load initial model from model_file.')
agent_host.addOptionalStringArgument('model_file', 'Path to the initial model file', '')
agent_host.addOptionalFlag('debug', 'Turn on debugging.')
agent_host.addOptionalIntArgument('x','The width of the arena.',18)
agent_host.addOptionalIntArgument('y','The width of the arena.',16)
agent_host.addOptionalIntArgument('items','Number of items to be spawned.',3)

malmoutils.parse_command_line(agent_host)

# -- set up the python-side drawing -- #
scale = 40
world_x = agent_host.getIntArgument('x')
world_y = agent_host.getIntArgument('y')
root = tk.Tk()
root.wm_title("Q-table")
canvas = tk.Canvas(root, width=world_x * scale, height=world_y * scale, borderwidth=0, highlightthickness=0, bg="black")
canvas.grid()
root.update()

num_items = agent_host.getIntArgument('items')

if agent_host.receivedArgument("test"):
    num_maps = 1
else:
    num_maps = 1

for imap in range(num_maps):

    # -- set up the agent -- #
    actionSet = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]

    agent = TabQAgent(
        actions=actionSet,
        epsilon=agent_host.getFloatArgument('epsilon'),
        alpha=agent_host.getFloatArgument('alpha'),
        gamma=agent_host.getFloatArgument('gamma'),
        debug=agent_host.receivedArgument("debug"),
        canvas=canvas,
        root=root)

    # -- set up the mission -- #
    mission_xml = XML_generator(x=world_x,y=world_y,items=num_items)
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
    my_mission.removeAllCommandHandlers()
    my_mission.allowAllDiscreteMovementCommands()
    my_mission.requestVideo(640, 480)
    my_mission.setViewpoint(1)

    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))  # add Minecraft machines here as available

    max_retries = 3
    agentID = 0
    expID = 'tabular_q_learning'

    num_repeats = 1600
    cumulative_rewards = []
    for i in range(num_repeats):

        print("\nMap %d - Mission %d of %d:" % (imap, i + 1, num_repeats))

        my_mission_record = malmoutils.get_default_recording_object(agent_host,
                                                                    "./save_%s-map%d-rep%d" % (expID, imap, i))

        for retry in range(max_retries):
            try:
                agent_host.startMission(my_mission, my_clients, my_mission_record, agentID, "%s-%d" % (expID, i))
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    print("here?")
                    time.sleep(2.5)

        print("Waiting for the mission to start", end=' ')
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
        print()

        # -- run the agent in the world -- #
        cumulative_reward = agent.run(agent_host)
        print('Cumulative reward: %d' % cumulative_reward)
        cumulative_rewards += [cumulative_reward]

        # -- clean up -- #
        time.sleep(0.5)  # (let the Mod reset)

    print("Done.")

    print()
    print("Cumulative rewards for all %d runs:" % num_repeats)
    print(cumulative_rewards)

    x_axis = [i + 1 for i in range(num_repeats)]
    plt.plot(x_axis, cumulative_rewards)
    plt.xlabel("iterations", fontsize = 16)
    plt.ylabel("Cumulative reward", fontsize = 16)
    plt.show()
    input()

from __future__ import print_function
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


from collections import namedtuple, deque, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import floor
from copy import deepcopy

if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk

save_images = False
if save_images:
    from PIL import Image

malmoutils.fix_print()


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # D_in = input dimension = 9 * 5: a five number encoding for each of the block types
        # plus a boolean for each block to denote if they have been visited
        # plus the previous six moves it has made so that it "perceive" movement
        # plus the 2 corrdinates of the agent's location
        # same reasoning as above but without the visited information
        #self.D_in = 9 * 8 + 2 + 2

        #D_in = input dimension = (vision(9) * length of block_list(8)) * (1 current state + 4 past states))
        self.D_in = (9 * 8) * (1+4)

        # D_out = output dimension = 4: 4 directions of move
        self.D_out = 4

        # H = hidden dimension, use a number between input and output dimension
        self.H = int((self.D_in + self.D_out) / 2)


        self.input_layer = nn.Linear(self.D_in, self.H)
        self.hidden_layer = nn.Linear(self.H, self.H)
        self.output_layer = nn.Linear(self.H, self.D_out)

    def forward(self, x):
#        print(x)
        h_relu = F.relu(self.input_layer(x))
#        print(h_relu)
        h_relu = F.relu(self.hidden_layer(h_relu))
#        print(h_relu)
        y_pred = self.output_layer(h_relu)
#        print(y_pred)
        return y_pred


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])

    def add(self, state, action, reward, next_state):
        """Add a new experience to memory."""
        exp = self.experience(state.detach().numpy(), action, reward, next_state.detach().numpy())
        self.memory.append(exp)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).float()#.to(deepQAgent.device)

        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).long()#.to(deepQAgent.device)

        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float()#.to(deepQAgent.device)

        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences if exp is not None])).float()#.to(deepQAgent.device)


        return (states, actions, rewards, next_states)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class deepQAgent(object):
    """Deep Q-learning agent for discrete state/action spaces."""
    def __init__(self, actions=[], learning_rate=0.1, tau=0.1, epsilon=1.0, gamma=0.99, debug=False, canvas=None, root=None, target_file=None, policy_file=None):

        self.block_list = ['sandstone', 'gold_block', 'red_sandstone', 'lapis_block',
                           'cobblestone', 'grass', 'lava', 'flowing_lava']               # all types of blocks agent can see
        self.buffer_size = int(1e5)             # replay buffer size
        self.batch_size = 32                     # minibatch size
        self.learning_rate = learning_rate      # learning rate
        self.tau = tau                          # for soft update of target parameters
        self.epsilon= epsilon                   # inital epsilon-greedy
        self.epsilon_decay = 0.00018#0.00009              # how quickly to decay epsilon
        self.gamma = gamma                      # discount factor
        self.update_every = 16                   # how often we updated the nn
        self.action_size = len(actions)
        self.movement_memory = 4

        # running PyTorch on cpu
        self.device = torch.device('cpu')

        # create network
        self.policy_model = DQN()#.to(self.device)
        self.target_model = deepcopy(self.policy_model)#DQN()#.to(self.device)
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=self.learning_rate)

        # if specified, read from target_file and policy_file
        if target_file is not None:
            self.target_model.load_state_dict(torch.load(target_file))
            self.target_model.eval()
            print("loaded target model from file!")
            input("Press Enter to acknowledge...")

        if policy_file is not None:
            self.policy_model.load_state_dict(torch.load(target_file))
            self.policy_model.eval()
            print("loaded policy model from file!")
            input("Press Enter to acknowledge...")

        # create memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        self.logger = logging.getLogger(__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.actions = actions

        self.canvas = canvas
        self.root = root

        self.rep = 0

        self.action_values_old = list()

        self.moves_temp = deque(maxlen=self.movement_memory*9)

        self.drawQ_reward_history = defaultdict(str)
        self.drawQ_map = defaultdict(str)

    def step(self, state, action, reward, next_state):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s') tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states = experiences

        # Get max predicted Q values (for next states) from target model
        next_action_targets = self.target_model(next_states)
        next_action = next_action_targets.max(1)[0].unsqueeze(-1)
        targets = rewards + (gamma * torch.Tensor(next_action))
#
        # Get expected Q values from policy model
        action_policy = self.policy_model(states)
        policy = action_policy.gather(1, actions)
        # Compute loss
        loss = F.mse_loss(policy, targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.policy_model, self.target_model, self.tau)

    def soft_update(self, policy_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_policy + (1 - τ)*θ_target
        """
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)

    def act(self, world_state, agent_host, old_obs=None):
        """Returns actions for given state as per current policy."""
        if old_obs is None:
            obs_text = world_state.observations[-1].text
            obs = json.loads(obs_text)  # most recent observation
        else:
            obs = old_obs

        self.logger.debug(obs)
        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        current_s = "%d,%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
        self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'ZPos'])))

        try:
            vision = obs['vision']
        except KeyError:
            vision = ["sandstone","sandstone","sandstone","sandstone","sandstone","sandstone","sandstone","sandstone","sandstone",] # 9 zeros

        encode = encode_observations(vision)

        emb = self.one_hot(torch.tensor(encode), len(self.block_list))

        input_state = torch.cat((emb.flatten(), self.moves))


        self.policy_model.eval()
        with torch.no_grad():
            action_values = self.policy_model(input_state)
            self.action_values_old = deepcopy(action_values)
        self.policy_model.train()


        self.drawQ( curr_x = int(obs[u'XPos']), curr_y = int(obs[u'ZPos']) , action_values = action_values)

        # Epsilon-greedy action selection
        _i = 0
        while True:
            if (_i < 1 and ((random.random() > self.epsilon) or test_knowledge)):
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = action_values.max(0)[0].max(0)[0].view(1, 1)
                action_list = list(action_values)
                action = torch.tensor([[action_list.index(action)]])
            else:
                action = torch.tensor([[random.randrange(self.action_size)]])
            if( (action == 0 and vision[1] != 'gold_block') or (action == 1 and vision[7] != 'gold_block') or
                (action == 2 and vision[3] != 'gold_block') or (action == 3 and vision[5] != 'gold_block') ):
                break
            else:
                print("abort (wall)")
                _i += 1
                
        self.moves_temp.extend(encode)
        self.moves = self.one_hot(torch.tensor(self.moves_temp), len(self.block_list)).flatten()

        # send the selected action
        agent_host.sendCommand(self.actions[action])
        self.prev_s = current_s
        self.prev_a = action

        if old_obs is None:
            obs_text = world_state.observations[-1].text
            obs = json.loads(obs_text)  # most recent observation
        else:
            obs = old_obs
        self.logger.debug(obs)
        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        current_s = "%d,%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
        self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'ZPos'])))

        return input_state, action

    def run(self, agent_host, test_knowledge):
        """Run agent on current world"""

        total_reward = 0
        current_r = 0
        tol = 0.01

        self.drawQ_reward_history = defaultdict(str)
        self.state = None
        self.action = None
        self.next_state = None
        
        for i in range(self.movement_memory*9): #9 for vision radius
            self.moves_temp.append(0)
        self.moves = self.one_hot(torch.tensor(self.moves_temp), len(self.block_list)).flatten()
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

        self.drawQ_reward_history[str(prev_x)+','+str(prev_z)] = 0.0

        if save_images:
            # save the frame, for debugging
            frame = world_state.video_frames[-1]
            image = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels))
            iFrame = 0
            self.rep = self.rep + 1
            image.save('rep_' + str(self.rep).zfill(3) + '_saved_frame_' + str(iFrame).zfill(4) + '.png')

        # take first action
        require_move = True
        check_expected_position = True

        count = 0

        # main loop:
        while world_state.is_mission_running:

            state, action = self.act(world_state, agent_host)
            #input("Press Enter to continue...")
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
            current_r += sum(r.getValue() for r in world_state.rewards)

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

                vision = obs['vision']
                encode = encode_observations(vision)

                emb = self.one_hot(torch.tensor(encode), len(self.block_list))
                next_state = torch.cat((emb.flatten(), self.moves))
                prev_x = curr_x
                prev_z = curr_z


                # place move into memory and update NN if necessary
                total_reward += current_r

                agent.step(state, action, total_reward, next_state)

                # save the total reward in the drawQ_reward_history dictionary
                # for drawQ
                self.drawQ_reward_history[str(curr_x)+","+str(curr_z)] = total_reward

                ### SPECIAL ###
                # Here, we can replace our current spot with a normal block
                # to indicate that the item has been picked up.
#                print(current_r)
                if obs[u'vision'][floor(len(obs[u'vision']) / 2)] == "grass":
                    temp = prev_x, prev_z
                    result = "chat /fill " + str(temp[0]) + " 45 " + str(temp[1]) + " " + str(temp[0]) + " 45 " + str(
                        temp[1]) \
                             + " minecraft:sandstone 0 replace minecraft:grass"
                    agent_host.sendCommand(result)

                ### END ###

                current_r = 0

                count += 1



        # process final reward
        self.logger.debug("Final reward: %d" % current_r)
        print("Final reward: %d" % current_r)
        total_reward += current_r

        x = obs[u'XPos']
        z = obs[u'ZPos']

        action = int(action)
        if action == 0:
            z -= 1
        elif action == 1:
            z += 1
        elif action == 2:
            x -= 1
        elif action == 3:
            x += 1

        obs[u'XPos'] = x
        obs[u'ZPos'] = z
        state, action = self.act(world_state, agent_host, old_obs=obs)
        agent.step(state, action, total_reward, state)
        self.drawQ_reward_history[str(obs[u'XPos']) + "," + str(obs[u'ZPos'])] = total_reward

        # update epsilon for next run but don't let epsilon get below 0.01
        if self.epsilon > 0.01 or True: #override for test
            self.epsilon -= self.epsilon_decay
        if self.epsilon < 0:
            self.epsilon = 0

        # stochastic means we must lower alpha to 0 over time
        #if self.learning_rate > 0:
        #    self.learning_rate -= 0.00003
        #if self.learning_rate < 0:
        #    self.learning_rate = 0

        #suggestion: raise gamma over time
        #if self.gamma < 0.99:
        #    self.gamma += 0.00003
        
        print()
        print('updated epsilon: ', self.epsilon)
        print('updated alpha:   ', self.learning_rate)
        print('updated gamma:   ', self.gamma)
        print()

        self.drawQ(curr_x = int(obs[u'XPos']), curr_y = int(obs[u'ZPos']))

        return total_reward

    def one_hot(self, batch, depth):
        emb = nn.Embedding(depth, depth)
        emb.weight.data = torch.eye(depth)
        return emb(batch)

    def drawQ( self, curr_x=None, curr_y=None ,action_values = None):
        if self.canvas is None or self.root is None:
            return
        self.canvas.delete("all")
        action_inset = 0.1
        action_radius = 0.1
        curr_radius = 0.02
        action_positions = [ ( 0.5, 1-action_inset ), ( 0.5, action_inset ), ( 1-action_inset, 0.5 ), ( action_inset, 0.5 ) ]
        #input("...")
        # (NSWE to match action order)
        for x in range(world_x):
            for y in range(world_y):
                block_color = self.drawQ_map[str(x)+","+str(y)] if self.drawQ_map[str(x)+","+str(y)] != "" else "#000"
                self.canvas.create_rectangle((world_x - 1 - x) * scale, (world_y - 1 - y) * scale,
                                             (world_x - 1 - x + 1) * scale, (world_y - 1 - y + 1) * scale,
                                             outline="#fff",
                                             fill=block_color)

                visited = str(x+0.5)+","+str(y+0.5) in self.drawQ_reward_history.keys()
                self.canvas.create_rectangle( (world_x-1-x)*scale, (world_y-1-y)*scale, (world_x-1-x+1)*scale, (world_y-1-y+1)*scale,
                                              outline="#ddd" if not visited else "#ddd",
                                              fill=None if not visited else "#000064")
                if visited:
                    self.canvas.create_text((world_x - 1 - x + 0.5) * scale,
                                            (world_y - 1 - y + 0.5) * scale,
                                            font=("Arial",6),
                                            text=str(self.drawQ_reward_history[str(x+0.5)+","+str(y+0.5)]),
                                            fill="#fff")


        if action_values is not None and curr_x is not None and curr_y is not None:
            x = curr_x
            y = curr_y
            min_value = min(action_values)
            max_value = max(action_values)
            for action in range(4):
                value = action_values[action]
                color = int( 255 * ( value - min_value ) / ( max_value - min_value )) # map value to 0-255
                color = max( min( color, 255 ), 0 ) # ensure within [0,255]
                color_string = '#%02x%02x%02x' % (255-color, color, 0)
                self.canvas.create_oval( (world_x - 1 - x + action_positions[action][0] - action_radius ) *scale,
                                         (world_y - 1 - y + action_positions[action][1] - action_radius ) *scale,
                                         (world_x - 1 - x + action_positions[action][0] + action_radius ) *scale,
                                         (world_y - 1 - y + action_positions[action][1] + action_radius ) *scale,
                                         outline=color_string, fill=color_string, )
        if curr_x is not None and curr_y is not None:
            self.canvas.create_oval( (world_x - 1 - curr_x + 0.5 - curr_radius ) * scale,
                                     (world_y - 1 - curr_y + 0.5 - curr_radius ) * scale,
                                     (world_x - 1 - curr_x + 0.5 + curr_radius ) * scale,
                                     (world_y - 1 - curr_y + 0.5 + curr_radius ) * scale,
                                     outline="#fff", fill="#fff" )

        self.root.update()

class XMLGenerator:
    def __init__(self, x, y):
        self.arena_width = x - 1
        self.arena_height = y
        self.used_pos = set()
        self.drawQ_map = defaultdict(str)
        self.reset()

    def reset(self):
        self.reset_used_pos()

    def reset_used_pos(self):
        self.used_pos = set()
        self.used_pos.add((self.arena_width, self.arena_height - 3))
        self.used_pos.add((self.arena_width, self.arena_height))

    def reset_drawQ_map(self):
        self.drawQ_map = defaultdict(str)

    def add_enemies(self, count=-1):
        arena_width = self.arena_width
        arena_height = self.arena_height
        used_pos = self.used_pos

        xml = ""
        # vision around enemies
        enemy_vision = set()

        if count <= -1:
            smaller_dim = min(arena_width, arena_height)
            enemies = (smaller_dim-1)//3
        else:
            enemies = count

        for i in range(enemies):
            while True:
                x = random.randint(2, arena_width)
                z = random.randint(0, arena_height-4)
                while (z <= 2 ) and x in range(3, 4+3):
                    x = random.randint(2, arena_width)
                    z = random.randint(0, arena_height-4)
                if (x,z) not in used_pos:
                    break

            used_pos.add((x,z))
            enemy_vision.add((x,z+1))
            enemy_vision.add((x,z+2))
            enemy_vision.add((x-1,z))
            enemy_vision.add((x-1,z+1))
            enemy_vision.add((x-1,z+2))
            enemy_vision.add((x-2,z))
            enemy_vision.add((x-2,z+1))
            enemy_vision.add((x-2,z+2))

            self.drawQ_map[str(x) + ',' + str(z)] = "#4e0000"
            self.drawQ_map[str(x) + ',' + str(z + 1)] = "#4e0000"
            self.drawQ_map[str(x) + ',' + str(z + 2)] = "#4e0000"
            self.drawQ_map[str(x - 1) + ',' + str(z)] = "#4e0000"
            self.drawQ_map[str(x - 1) + ',' + str(z + 1)] = "#4e0000"
            self.drawQ_map[str(x - 1) + ',' + str(z + 2)] = "#4e0000"
            self.drawQ_map[str(x - 2) + ',' + str(z)] = "#4e0000"
            self.drawQ_map[str(x - 2) + ',' + str(z + 1)] = "#4e0000"
            self.drawQ_map[str(x - 2) + ',' + str(z + 2)] = "#4e0000"

            xml += '''<DrawCuboid x1="''' + str(x) + '''" y1="45" z1="''' + str(z) + '''" x2="''' + str(x-2) + '''" y2="45" z2="''' + str(z+2) + '''" type="red_sandstone"/>'''
            xml += '''<DrawEntity x="''' + str(x-0.5) + '''" y="45" z="''' + str(z+1.5) + '''"  type="Villager" />'''

        used_pos.update(enemy_vision)

        self.used_pos = used_pos
        return xml

    def add_items(self, items_count=-1):
        arena_width = self.arena_width
        arena_height = self.arena_height
        used_pos = self.used_pos

        xml = ""
        #print(used_pos)

        if items_count == -1:
            # Use algorithm
            smaller_dim = min(arena_width, arena_height)
            items_count = (smaller_dim - 1) // 3

        for i in range(items_count):
            while True:
                x = random.randint(0, arena_width)
                z = random.randint(0, arena_height - 1)
                if (x,z) not in used_pos:
                    break
            used_pos.update((x, z))
            self.drawQ_map[str(x)+','+str(z)] = "#003b00"

            xml += '''<DrawItem x="''' + str(x) + '''" y="46" z="''' + str(z) + '''" type="diamond" />''' + \
                '''<DrawBlock x="''' + str(x) + '''" y="45" z="''' + str(z) + '''" type="grass" />'''

        self.used_pos = used_pos
        return xml

    def add_random_walls_and_lava(self, count=1):
        arena_width = self.arena_width
        arena_height = self.arena_height
        used_pos = self.used_pos

        xml = ""
        #print(used_pos)

        for i in range(count):

            while True:
                x = random.randint(0, arena_width)
                z = random.randint(0, arena_height - 1)
                if (x, z) not in used_pos:
                    break
            used_pos.update((x, z))
            xml += '''<DrawBlock x="''' + str(x) + '''" y="45" z="''' + str(z) + '''" type="'''+\
                   ("lava" if random.randint(0,1) else "lava") + '''" />'''
            self.drawQ_map[str(x)+','+str(z)] = "#4e0000"
        self.used_pos = used_pos
        return xml

    def set_goal(self, x=None,y=None):
        arena_width = self.arena_width
        arena_height = self.arena_height
        used_pos = self.used_pos

        xml = ""
        print(used_pos)

        z = y
        while (x is None or z is None) or (x,z) in used_pos:
            x = random.randint(0, arena_width)
            z = random.randint(arena_height-2, arena_height - 1)
            #input("x = "+str(x)+", y = "+str(y)+"...")

        used_pos.update((x, z))
        used_pos.add((x, z + 1))
        used_pos.add((x, z + 2))
        used_pos.add((x - 1, z))
        used_pos.add((x - 1, z + 1))
        used_pos.add((x - 1, z + 2))
        used_pos.add((x - 2, z))
        used_pos.add((x - 2, z + 1))
        used_pos.add((x - 2, z + 2))

        self.drawQ_map[str(x) + ',' + str(z)] = "#00ffff"

        xml += '''<DrawItem x="''' + str(x) + '''" y="46" z="''' + str(z) + '''" type="diamond" />''' + \
            '''<DrawBlock x="''' + str(x) + '''" y="45" z="''' + str(z) + '''" type="lapis_block" />'''

        self.used_pos = used_pos
        return xml

    def get_drawQ_map(self):
        return self.drawQ_map

    def XML_generator(self):
        arena_width = self.arena_width
        arena_height = self.arena_height
        used_pos = self.used_pos

        # make sure nothing spawns on top of starting position
        used_pos.add((4, 1))

        xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    
                      <About>
                        <Summary>Avoiding enemies to get to target.</Summary>
                      </About>
    
                      <ModSettings>
                        <MsPerTick>40</MsPerTick>
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
                              <DrawCuboid x1="-2" y1="46" z1="-2" x2="''' + str(
            arena_width + 2) + '''" y2="50" z2="''' + str(arena_height + 2) + '''" type="air" />            <!-- limits of our arena -->
                              <DrawCuboid x1="-2" y1="45" z1="-2" x2="''' + str(
            arena_width + 2) + '''" y2="45" z2="''' + str(arena_height + 2) + '''" type="lava" />           <!-- lava floor -->
                              <DrawCuboid x1="-1"  y1="44" z1="0"  x2="''' + str(arena_width) + '''" y2="45" z2="''' + str(
            arena_height) + '''" type="sandstone" />      <!-- floor of the arena -->
    
                              <DrawBlock  x="4"   y="45"  z="1"  type="cobblestone" />                           <!-- the starting marker -->
    
                              <!-- Boundary -->
                              <DrawCuboid x1="''' + str(arena_width + 1) + '''"  y1="45" z1="-1"  x2="''' + str(
            arena_width + 1) + '''" y2="45" z2="''' + str(arena_height) + '''" type="gold_block" />           <!-- Left wall from start position -->
                              <DrawCuboid x1="-1"  y1="45" z1="-1"  x2="''' + str(arena_width + 1) + '''" y2="45" z2="-1" type="gold_block" />			  <!-- Bottom wall from start position -->
                              <DrawCuboid x1="-1"  y1="45" z1="-1"  x2="-1" y2="45" z2="''' + str(arena_height) + '''" type="gold_block" />           <!-- Right wall from start position -->
                              <DrawCuboid x1="-1"  y1="45" z1="''' + str(arena_height) + '''"  x2="''' + str(
            arena_width + 1) + '''" y2="45" z2="''' + str(arena_height) + '''" type="gold_block" />           <!-- Top wall from start position -->
    
                              
                                  
                              <!-- Enemies -->
                              ''' + self.add_enemies(agent_host.getIntArgument('fenemies')) + '''
    
                              <!-- Items -->
                              ''' + self.add_items(agent_host.getIntArgument('items')) + '''
    
                              <!-- Extra Walls and Lava -->
                              ''' + self.add_random_walls_and_lava(agent_host.getIntArgument('obstacles')) + '''
                              
                              <!-- Goal -->
                              ''' + self.set_goal(x=arena_width,y=arena_height-1) + '''
    
                          </DrawingDecorator>
                          <ServerQuitFromTimeUp timeLimitMs="50000"/>
                          <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                      </ServerSection>
    
                      <AgentSection mode="Survival">
                        <Name>Master</Name>
                        <AgentStart>
                          <Placement x="4.5" y="46.0" z="1.5" pitch="80" yaw="0"/>
                          <Inventory>
                          </Inventory>
                        </AgentStart>
                        <AgentHandlers>
                          <ObservationFromFullStats/>
                          <ChatCommands/>
                          <ObservationFromChat/>
                          <ObservationFromGrid>
                              <Grid name="vision">
                                <min x="-1" y="-1" z="-1"/>
                                <max x="1" y="-1" z="1"/>
                              </Grid>
                          </ObservationFromGrid>
                          <VideoProducer want_depth="false">
                              <Width>160</Width>
                              <Height>120</Height>
                          </VideoProducer>
                          <DiscreteMovementCommands>
                              <ModifierList type="deny-list">
                                <command>attack</command>
                              </ModifierList>
                          </DiscreteMovementCommands>
                          <RewardForTouchingBlockType>
                            <Block reward="-10000.0" type="lava" behaviour="onceOnly"/>
                            <Block reward="1000.0" type="lapis_block" behaviour="onceOnly"/>
                            <Block reward="-100.0" type="red_sandstone" behaviour="onceOnly"/>
                            <Block reward="0.0" type="gold_block"/>
                            <Block reward="20" type="grass" />
                          </RewardForTouchingBlockType>
                          <RewardForSendingCommand reward="-1"/>
                          <AgentQuitFromTouchingBlockType>
                              <Block type="lava" />
                              <Block type="lapis_block" />
                              <Block type="red_sandstone" />
                              <Block type="stone" />
                          </AgentQuitFromTouchingBlockType>
                        </AgentHandlers>
                      </AgentSection>
    
                    </Mission>'''
        self.used_pos = used_pos
        return xml


def encode_observations(vision:list=list()):
    encode_dict = defaultdict(lambda: 0,
        sandstone = 0,
        flowing_lava = 1,
        lapis_block = 2,
        red_sandstone = 3,
        stone = 4,
        gold_block = 5,
        grass = 6,
        cobblestone = 7,
        lava = 1,
                              )

    result = []
    for item in vision:
        result.append(encode_dict[item])
    return result


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
agent_host.addOptionalFloatArgument('tau',
                                    'Soft update of target parameters.', 0.1)
agent_host.addOptionalFloatArgument('epsilon',
                                    'Exploration rate of the Q-learning agent.', 1.0)
agent_host.addOptionalFloatArgument('gamma', 'Discount factor.', 0.99)
agent_host.addOptionalFlag('load_model', 'Load initial model from model_file.')
agent_host.addOptionalStringArgument('model_file', 'Path to the initial model file', '')
agent_host.addOptionalFlag('debug', 'Turn on debugging.')
agent_host.addOptionalIntArgument('x','The width of the arena.',25)
agent_host.addOptionalIntArgument('y','The width of the arena.',25)
agent_host.addOptionalIntArgument('items','The total number of small items in the arena (except the goal)', 10)
agent_host.addOptionalIntArgument('obstacles','The total number of extra walls/lava, for interest', 0)
agent_host.addOptionalIntArgument('fenemies','The total number of enemies', -1)
agent_host.addOptionalStringArgument('policy_file', 'Load policy model from path','')
agent_host.addOptionalStringArgument('target_file', 'Load target model from path','')
malmoutils.parse_command_line(agent_host)

# -- set up the python-side drawing -- #
scale = 30
world_x = agent_host.getIntArgument('x')
world_y = agent_host.getIntArgument('y')
root = tk.Tk()
root.wm_title("Q-table")
canvas = tk.Canvas(root, width=world_x * scale, height=world_y * scale, borderwidth=0, highlightthickness=0, bg="black")
canvas.grid()
root.update()

if agent_host.receivedArgument("test"):
    num_maps = 1
else:
    num_maps = 30000

try:
    for imap in range(num_maps):

        # -- set up the agent -- #
        actionSet = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]

        agent = deepQAgent( actions=actionSet,
                            learning_rate=agent_host.getFloatArgument('alpha'),
                            tau=agent_host.getFloatArgument('tau'),
                            epsilon=agent_host.getFloatArgument('epsilon'),
                            gamma=agent_host.getFloatArgument('gamma'),
                            debug=agent_host.receivedArgument("debug"),
                            canvas=canvas,
                            root=root,
                            target_file=agent_host.getStringArgument("target_file") if len(agent_host.getStringArgument("target_file")) > 0 else None,
                            policy_file=agent_host.getStringArgument("policy_file") if len(agent_host.getStringArgument("policy_file")) > 0 else None)




        max_retries = 3
        agentID = 0
        expID = 'deep_q_learning'

        num_repeats = 15000
        cumulative_rewards = []
        for i in range(num_repeats):

            # -- set up the mission -- #
            xml_object = XMLGenerator(x=world_x,y=world_y)
            mission_xml= xml_object.XML_generator()
            agent.drawQ_map = xml_object.get_drawQ_map()
            my_mission = MalmoPython.MissionSpec(mission_xml, True)
            my_mission.removeAllCommandHandlers()
            my_mission.allowAllChatCommands()
            my_mission.allowAllDiscreteMovementCommands()
            my_mission.requestVideo(320, 240)
            my_mission.setViewpoint(1)
    
            my_clients = MalmoPython.ClientPool()
            my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))  # add Minecraft machines here as available

            test_knowledge = True if i % 5 == 0 else False

            print("\nMap %d - Mission %d of %d:" % (imap, i, num_repeats))
            root.wm_title("Mini-map "+"#%d/%d %s:" % (i, num_repeats,
                                                                         "(TEST POLICY)" if test_knowledge else ""))
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


            cumulative_reward = agent.run(agent_host, test_knowledge)
            print('Cumulative reward: %d' % cumulative_reward)
            cumulative_rewards += [cumulative_reward]


            # -- clean up -- #
            time.sleep(1)  # (let the Mod reset)

        print("Done.")

        print()
        print("Cumulative rewards for all %d runs:" % num_repeats)
        print(cumulative_rewards)
except KeyboardInterrupt as e:
    print("KeyboardInterrupt detected: aborting...")
finally:
    print("Saving model file to same directory...")
    torch.save(agent.policy_model.state_dict(), "./policy_model.pth")
    torch.save(agent.target_model.state_dict(), "./target_model.pth")
    print("Saved model.pth file!")

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


from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import floor
from copy import deepcopy

enemies = 2


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
        self.D_in = 9 * 8 + 9 + 6 + 2

        # H = hidden dimension, use a number between input and output dimension
        self.H = 50
        # D_out = output dimension = 4: 4 directions of move

        self.D_out = 4
        
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

#        print()
#        print()
#        print("experiences")
#        print(experiences)
#        print()
#        print("sampling")
        
#        print(np.vstack([exp.state for exp in experiences if exp is not None]))
        states = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).float()#.to(deepQAgent.device)
        
#        print(np.vstack([exp.action for exp in experiences if exp is not None]))
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).long()#.to(deepQAgent.device)
        
#        print(np.vstack([exp.reward for exp in experiences if exp is not None]))
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float()#.to(deepQAgent.device)
        
#        print(np.vstack([exp.next_state for exp in experiences if exp is not None]))
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences if exp is not None])).float()#.to(deepQAgent.device)

        
        return (states, actions, rewards, next_states)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    

class deepQAgent(object):
    """Deep Q-learning agent for discrete state/action spaces."""
    def __init__(self, actions=[], learning_rate=0.1, tau=0.1, epsilon=1.0, gamma=0.99, debug=False, canvas=None, root=None):
        
        self.block_list = ['sandstone', 'gold_block', 'red_sandstone', 'lapis_block',
                           'cobblestone', 'grass', 'lava', 'flowing_lava']               # all types of blocks agent can see
        self.buffer_size = int(1e5)             # replay buffer size
        self.batch_size = 64                    # minibatch size
        self.learning_rate = learning_rate      # learning rate
        self.tau = tau                          # for soft update of target parameters
        self.epsilon= epsilon                   # inital epsilon-greedy
        self.epsilon_decay = 0.00009              # how quickly to decay epsilon
        self.gamma = gamma                      # discount factor
        self.update_every = 8                   # how often we updated the nn
        self.action_size = len(actions)
        self.movement_memory = 6
        
        # running PyTorch on cpu
        self.device = torch.device('cpu')
        
        # create network
        self.policy_model = DQN()#.to(self.device)
        self.target_model = DQN()#.to(self.device)
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=self.learning_rate)
        
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
    
    def step(self, state, action, reward, next_state):
#        print()
#        print(state)
#        print(next_state)
#        print()
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

#        print(actions.size())
#        print(next_states.size())
        # Get max predicted Q values (for next states) from target model
        next_action_targets = self.target_model(next_states)
#        print(next_action_targets.size())
#        print(next_action_targets)
#        print(next_action_targets.max(1)[0].unsqueeze(-1))
#        print(np.average(next_action_targets.cpu().data.numpy(),axis=1))
        next_action = next_action_targets.max(1)[0].unsqueeze(-1)
#        next_action = next_action_targets.max(1)[0].max(1)[0]
#        next_action = np.max((np.average(next_action_targets.cpu().data.numpy(),axis=1)),axis=1)
#        print("next action")
#        print(next_action)
#        print(len(next_action))
#        next_targets = self.target_model(next_states).detach().max(1)[0]
#        print(next_targets)
#        print(next_targets.size())
#        next_targets = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
#        print(next_targets)
#        print(next_targets.size())
        # Compute Q targets for current states 
#        print("gamma")
#        print(gamma)
#        print("next_action")
#        print(next_action)
#        print(next_action.size())
#        print("rewards")
#        print(rewards.size())
#        print(rewards)
#        print(gamma * next_action)
        targets = rewards + (gamma * torch.Tensor(next_action))
#        print(torch.Tensor(next_action))
#        print(gamma * torch.Tensor(next_action))
#        print((gamma * torch.Tensor(next_action)))
#        print((gamma * torch.Tensor(next_action)).size())
#        print(rewards.size())
#        print("targets")
#        print(targets)
#        print(targets.size())
#        
        # Get expected Q values from policy model
#        print("actions")
#        print(actions)
        action_policy = self.policy_model(states)
#        actions = actions.expand_as(action_prob)
#        expected_q = action_prob.gather(1, actions)
        policy = action_policy.gather(1, actions)
#        policy = action_policy.max(1)[0].max(1)[0].unsqueeze(1)
#        policy = torch.Tensor(np.max((np.average(action_policy.cpu().data.numpy(),axis=1)),axis=1)).unsqueeze(1)
#        expected_q = self.policy_model(states).gather(1, actions)
#        print(policy.size())
#        print(policy)
#        
#        print(policy)
#        print(policy.size())
#        print(targets)
#        print(targets.size())
        # Compute loss
        loss = F.mse_loss(policy, targets)
#        print(loss)
#        print(prev)
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
#        print()
#        print('target params: ', target_model.parameters())
#        print('local params: ', policy_model.parameters())
#        print()
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)
            
    def act(self, world_state, agent_host):
        """Returns actions for given state as per current policy."""
        visits = list()
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text)  # most recent observation
        #print(obs)
        self.logger.debug(obs)
        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        current_s = "%d,%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
        self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'ZPos'])))
        
#        print()
#        print("before command")
        print(obs)
        
        xpos = obs[u'XPos']
        zpos = obs[u'ZPos']
        
        curr_x = xpos
        curr_z = zpos
        
        while True:
            try:
                vision = obs['vision']
                break
            except KeyError:
                continue
        encode = list()
        for block in vision:
            encode.append(self.block_list.index(block))

#        input_state = self.one_hot(torch.tensor(encode), len(encode)).flatten()       .float()
#        print(input_state)
        visits = list()
#        print(self.visited)
        for surrounding in [(-1,-1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1,1)]:
            if (int(curr_x) + surrounding[0], int(curr_z) + surrounding[1]) not in self.visited:
                visits.append(0)
            else: 
                visits.append(1)
        
        emb = self.one_hot(torch.tensor(encode), len(self.block_list))
#        print(emb)
#        input_state = emb.flatten()
#        print(input_state)

#        emb_np = emb.detach().numpy()
#        print()
#        print(emb)
#        print(emb.flatten())
#        print(torch.cat((emb.flatten(), torch.as_tensor(visits).float())))
#        input_state = torch.cat((emb.flatten(), torch.as_tensor(visits).float()))
        input_state = torch.cat((torch.cat((torch.cat((emb.flatten(),
                                                       torch.as_tensor(visits).float())),
                                            torch.as_tensor(self.moves).float())),
                                 torch.tensor([int(curr_x), int(curr_z)]).float()))
#        print(input_state)
#        print(emb_np)
#        print()
#        
#            
#        state = np.array([int(xpos),
#                          int(zpos),
#                          int(visit_code)])
#        
#        state.append(emb_np)
    
        # update Q values        
#        state = torch.from_numpy(state).float().unsqueeze(0)#.to(self.device)
#        print(state)
#        print()
        
#        print(self.visited)
#        for surrounding in [(-1,-1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1,1)]:
#            if (int(xpos) + surrounding[0], int(zpos) + surrounding[1]) not in self.visited:
#                visits.append(0.)
#            else: 
#                visits.append(1.)
                
#        print(visits)
#        visit_tensor = torch.as_tensor(visits).unsqueeze(0)
#        print(visit_tensor)
#        print(input_state)
#        print(torch.cat((input_state, visit_tensor)))
#        print(torch.cat((input_state, visit_tensor)).t())
#        print(torch.cat((input_state, visit_tensor)).t().unsqueeze(0))
#        input_state = torch.cat((input_state, visit_tensor)).t().unsqueeze(0)
        
#        state_actions = torch.as_tensor(self.recent_actions, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#        print(state_actions.size())
#        print(input_state.size())
#        print(torch.cat((input_state, state_actions), dim=1))
#        print(torch.cat((input_state, state_actions), dim=1).t())
#        input_state = torch.cat((input_state, state_actions), dim=1)
#        print(input_state)
        

        self.policy_model.eval()
        with torch.no_grad():
            action_values = self.policy_model(input_state)
        self.policy_model.train()
        
#        print(state)
#        print(action_values)


        # Epsilon-greedy action selection
#        print(action_values)
#        print(action_values.max(1))
#        print(action_values.max(1)[0].max(1)[1])
#        print(action_values.max(1)[0].max(1)[1].view(1, 1))
#        print(np.average(action_values.cpu().data.numpy(),axis=1))
#        print(np.argmax(np.average(action_values.cpu().data.numpy(),axis=1)))

        _i = 0
        print("Test Knowledge: "+str(test_knowledge))
        while True:
            if ((random.random() > self.epsilon) or (test_knowledge and _i < 5)):
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                print("Optimal action: ", end="")
                action = action_values.max(0)[0].max(0)[0].view(1, 1)
#                action = np.average(action_values.cpu().data.numpy(),axis=1)
#                action = np.argmax(action_values.cpu().data.numpy())
            else:
                print("Random action: ", end="")
                action = torch.tensor([[random.randrange(self.action_size)]])
#                action = random.choice(np.arange(self.action_size))
            if( (action == 0 and vision[1] != 'gold_block') or (action == 1 and vision[7] != 'gold_block') or
                (action == 2 and vision[3] != 'gold_block') or (action == 3 and vision[5] != 'gold_block') ):
                break
            else:
                print("abort (wall)")
            _i += 1
        
        self.moves.append(action)
#        self.recent_actions.append(action)
#        print(self.recent_actions)
#        print(input_state)
#        print(np.argmax(action_values.cpu().data.numpy()))
        print(action)
#        print(np.average(action_values.cpu().data.numpy(),axis=1))
#        print(np.argmax(np.average(action_values.cpu().data.numpy(),axis=1)))
#        print(self.actions)
#        pr int(prev)

        # send the selected action
        agent_host.sendCommand(self.actions[action])
        self.prev_s = current_s
        self.prev_a = action

        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text)  # most recent observation
#        print()
#        print("after command")
#        print(obs)
        self.logger.debug(obs)
        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        current_s = "%d,%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
        self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'ZPos'])))
            
#        print(next_state)
#        print()
#        print()
            
        return input_state, action
    
    def run(self, agent_host, test_knowledge):
        """Run agent on current world"""

        total_reward = 0
        current_r = 0
        tol = 0.01

#        self.prev_s = None
#        self.prev_a = None
        self.state = None
        self.action = None
        self.next_state = None
        
        self.visited = set() # always have starting position set to visited
        self.visited.add((4, 1))
        self.moves = deque(maxlen=self.movement_memory)
        for i in range(self.movement_memory):
            self.moves.append(-1)
        

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
#        print(pre_state)
#        print(state)
        
#        obs = json.loads(world_state.observations[-1].text)
#        print("after first action")
#        print("curr_location: ", obs[u'XPos'], obs[u'ZPos'])
        

        require_move = True
        check_expected_position = True
        
        count = 0

        # main loop:
        while world_state.is_mission_running:
            
            state, action = self.act(world_state, agent_host)
            
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
                encode = list()
                for block in vision:
                    encode.append(self.block_list.index(block))
#                input_state = self.one_hot(torch.tensor(encode), len(encode)).flatten()       .float()
#                print(input_state)
                visits = list()
#                print(self.visited)
#                print(self.visited)
                if (int(curr_x), int(curr_z)) not in self.visited:
                    discovery_reward = 5
                    self.visited.add((int(curr_x), int(curr_z)))
                else:
                    discovery_reward = 0
#                print(self.visited)
                for surrounding in [(-1,-1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1,1)]:
                    if (int(curr_x) + surrounding[0], int(curr_z) + surrounding[1]) not in self.visited:
                        visits.append(0)
                    else: 
                        visits.append(1)
                
                emb = self.one_hot(torch.tensor(encode), len(self.block_list))
#                print(emb)
#                next_state = emb.flatten()
#                print(next_state)
#                next_state = torch.cat((emb.flatten(), torch.as_tensor(visits).float()))
                next_state = torch.cat((torch.cat((torch.cat((emb.flatten(),
                                                              torch.as_tensor(visits).float())),
                                                   torch.as_tensor(self.moves).float())),
                                        torch.tensor([int(curr_x), int(curr_z)]).float()))
#                print(next_state)
#                state_info = list()
#                state_info.append(vision)
#                encode = list()
#                for block in vision:
#                    encode.append(self.block_list.index(block))
#                print(self.block_list)
#                print(encode)
#                print(self.one_hot(torch.tensor(encode), len(self.block_list)).float())
#                next_state = self.one_hot(torch.tensor(encode), len(self.block_list)).t().float()
                
                
#                visits = list()
##                print(self.visited)
#                if (int(curr_x), int(curr_z)) not in self.visited:
#                    discovery_reward = 5
#                    self.visited.add((int(curr_x), int(curr_z)))
#                else:
#                    discovery_reward = 0
##                print(self.visited)
#                for surrounding in [(-1,-1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1,1)]:
#                    if (int(curr_x) + surrounding[0], int(curr_z) + surrounding[1]) not in self.visited:
#                        visits.append(0)
#                    else: 
#                        visits.append(1)
#                
#                state_info.append(visits)
#                
#                print(state_info)
                
#                visit_tensor = torch.as_tensor(visits).unsqueeze(0)
#                next_state = torch.cat((next_state, visit_tensor)).t().unsqueeze(0)
#                print(state)
#                print(action)
#                print(current_r)
                current_r += discovery_reward
#                print(current_r)
#                print(next_state)
#                
#                if count == 10:
#                    print(prev)
                
                prev_x = curr_x
                prev_z = curr_z


                # place move into memory and update NN if necessary
                total_reward += current_r

                agent.step(state, action, current_r, next_state)

                ### SPECIAL ###
                # Here, we can replace our current spot with a normal block
                # to indicate that the item has been picked up.
                print(current_r)
                # if current_r == 99:  # Reward of grass is 100 - 1
                if obs[u'vision'][floor(len(obs[u'vision']) / 2)] == "grass":
                    # my_mission.drawBlock(int(obs[u'XPos']),45,int(obs[u'ZPos']),"sandstone")
                    #temp = self.prev_s.split(",")
                    temp = prev_x, prev_z
                    # print(temp)
                    # print("grass detected via reward!")
                    result = "chat /fill " + str(temp[0]) + " 45 " + str(temp[1]) + " " + str(temp[0]) + " 45 " + str(
                        temp[1]) \
                             + " minecraft:sandstone 0 replace minecraft:grass"
                    # print(result)
                    # agent_host.sendCommand('chat /kill @a')
                    agent_host.sendCommand(result)
                    # agent_host.sendCommand("turn 1")
                    # input()

                ### END ###


#                print("current reward for this step: ", current_r)
#                print("total reward for this step: ", total_reward)
#                time.sleep(0.5)

                current_r = 0
                
                count += 1



        # process final reward
        self.logger.debug("Final reward: %d" % current_r)
        total_reward += current_r
        
        state = np.array([int(obs[u'XPos']),
                          int(obs[u'ZPos'])])
        # update Q values
        state = torch.from_numpy(state).float().unsqueeze(0)#.to(self.device)
        self.policy_model.eval()

        # update epsilon for next run but don't let epsilon get below 0.01
        if self.epsilon > 0.01 or True: #override for test
            self.epsilon -= self.epsilon_decay
        if self.epsilon < 0:
            self.epsilon = 0
        print()
        print('updated epsilon: ', self.epsilon)
        print()

        return total_reward
    
    def one_hot(self, batch, depth):
        emb = nn.Embedding(depth, depth)
        emb.weight.data = torch.eye(depth)
        return emb(batch)
    
    def emb(self, state_info):
        emb = nn.Embedding(len(state_info[0]), 1)
        return emb(state_info)

def add_enemies(arena_width,arena_height):
    xml = ""
    # add more enemies, but avoid end goal
    used_pos = set((arena_width, arena_height-3))
    
    smaller_dim = min(arena_width, arena_height)
    enemies = (smaller_dim-1)//3
    
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
        xml += '''<DrawCuboid x1="''' + str(x) + '''" y1="45" z1="''' + str(z) + '''" x2="''' + str(x-2) + '''" y2="45" z2="''' + str(z+2) + '''" type="red_sandstone"/>'''
        xml += '''<DrawEntity x="''' + str(x-0.5) + '''" y="45" z="''' + str(z+1.5) + '''"  type="Villager" />'''
    return xml


def add_items(arena_width, arena_height, items_count=1):
    xml = ""
    for i in range(items_count):

        if True:
            x = random.randint(0, arena_width)
            z = random.randint(0, arena_height - 1)

        xml += '''<DrawItem x="''' + str(x) + '''" y="46" z="''' + str(z) + '''" type="diamond" />''' + \
            '''<DrawBlock x="''' + str(x) + '''" y="45" z="''' + str(z) + '''" type="grass" />'''
    return xml

def encode_observations(vision:list=list()):
    """CURRENTLY NOT USED!!! block_list used instead for encoding."""
    encode_dict = {
        "sandstone": 0,
        "flowing_lava": 1,
        "lapis_block": 2,
        "red_sandstone": 3,
        "stone": 4,
        "gold_block": 5,
        "grass": 6,
        "cobblestone": 7,
        "lava": 1,
    }

    result = []
    for item in vision:
        result.append(encode_dict[item])
    print(str(vision[0:3])+"\n"+str(vision[3:6])+"\n"+str(vision[6:9]))
    print(str(result[0:3])+"\n"+str(result[3:6])+"\n"+str(result[6:9]))
    return result

def XML_generator(x,y):
    arena_width=x-1
    arena_height=y
    print(x,y)
    xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                
                  <About>
                    <Summary>Avoiding enemies to get to target.</Summary>
                  </About>
                  
                  <ModSettings>
                    <MsPerTick>50</MsPerTick>
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
                
                          <DrawBlock  x="''' + str(arena_width) + '''"   y="45"  z="''' + str(arena_height-1) + '''" type="lapis_block" />                           <!-- the destination marker -->
                          <DrawItem   x="''' + str(arena_width) + '''"   y="46"  z="''' + str(arena_height-1) + '''" type="diamond" />                               <!-- another destination marker -->
                          <DrawItem   x="6"   y="46"  z="0" type="diamond" />  
                		  
                          <!-- Enemies -->
                          '''+ add_enemies(arena_width, arena_height) + '''
                          
                          <!-- Items -->
                          '''+ add_items(arena_width, arena_height, agent_host.getIntArgument('i')) + '''
                		  
                      </DrawingDecorator>
                      <ServerQuitFromTimeUp timeLimitMs="30000"/>
                      <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                  </ServerSection>
                
                  <AgentSection mode="Survival">
                    <Name>Master</Name>
                    <AgentStart>
                      <Placement x="4.5" y="46.0" z="1.5" pitch="70" yaw="0"/>
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
                          <Width>640</Width>
                          <Height>480</Height>
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
                        <Block reward="100" type="grass" />
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
agent_host.addOptionalFloatArgument('tau',
                                    'Soft update of target parameters.', 0.1)
agent_host.addOptionalFloatArgument('epsilon',
                                    'Exploration rate of the Q-learning agent.', 1.0)
agent_host.addOptionalFloatArgument('gamma', 'Discount factor.', 0.99)
agent_host.addOptionalFlag('load_model', 'Load initial model from model_file.')
agent_host.addOptionalStringArgument('model_file', 'Path to the initial model file', '')
agent_host.addOptionalFlag('debug', 'Turn on debugging.')
agent_host.addOptionalIntArgument('x','The width of the arena.',18)
agent_host.addOptionalIntArgument('y','The width of the arena.',16)
agent_host.addOptionalIntArgument('i','The total number of small items in the arena (except the goal)', 5)

malmoutils.parse_command_line(agent_host)

# -- set up the python-side drawing -- #
scale = 50
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

for imap in range(num_maps):

    # -- set up the agent -- #
    actionSet = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]

#    agent = TabQAgent(
#        actions=actionSet,
#        epsilon=agent_host.getFloatArgument('epsilon'),
#        alpha=agent_host.getFloatArgument('alpha'),
#        gamma=agent_host.getFloatArgument('gamma'),
#        debug=agent_host.receivedArgument("debug"),
#        canvas=canvas,
#        root=root)
    
    agent = deepQAgent( actions=actionSet,
                        learning_rate=agent_host.getFloatArgument('alpha'),
                        tau=agent_host.getFloatArgument('tau'),
                        epsilon=agent_host.getFloatArgument('epsilon'),
                        gamma=agent_host.getFloatArgument('gamma'),
                        debug=agent_host.receivedArgument("debug"),
                        canvas=canvas,
                        root=root)
    


    # -- set up the mission -- #
    mission_xml = XML_generator(x=world_x,y=world_y)
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
    my_mission.removeAllCommandHandlers()
    my_mission.allowAllChatCommands()
    my_mission.allowAllDiscreteMovementCommands()
    my_mission.requestVideo(640, 480)
    my_mission.setViewpoint(1)

    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))  # add Minecraft machines here as available

    max_retries = 3
    agentID = 0
    expID = 'deep_q_learning'

    num_repeats = 15000
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
        test_knowledge = True if i % 5 == 0 else False

        cumulative_reward = agent.run(agent_host, test_knowledge)
        print('Cumulative reward: %d' % cumulative_reward)
        cumulative_rewards += [cumulative_reward]

        # -- clean up -- #
        time.sleep(2)  # (let the Mod reset)

    print("Done.")

    print()
    print("Cumulative rewards for all %d runs:" % num_repeats)
    print(cumulative_rewards)

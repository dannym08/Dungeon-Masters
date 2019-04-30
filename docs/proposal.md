---
layout: default
title:  Proposal
---

# Summary
----------
Our goal in this project is to create an agent that will train to solve a dungeon. These dungeons will be represented by rooms, which will be basic at first, with an exit and entrance. As we make progress we will add complexity such as: 
- enemies that the agent must avoid 
- walls that the agent must walk around

Our dungeons will initially consist of one floor. We eventually want to add a third dimension to the dungeon by creating a ladder up to a higher floor that is bigger and more difficult but offers a greater reward. 

The input will be a dungeon map represented in XML following the Malmo specifications. The output will be a calculated score that is described in more detail in the evaluation plan.

This project’s agent may be applicable to other maze solver games. It could theoretically work with physical agents in the real world: if we set a tiny wheeled robot in a room, this project could be used to help map its environment and move the robot around.

# AI/ML Algorithm
-----------------
We will be using reinforcement learning (with variations of either the Markov chains or neuron networks or both) to teach the agent to avoid obstacles and find the exit.

# Evaluation Plan (before meeting with Professor Singh)
------------------
Our basic metric for evaluation will be the number of steps that the agent takes to reach the exit of the dungeon. This metric will be measured by a score/points variable that decreases the more steps it takes. The agent will lose many points by dying to dungeon hazards (ie. enemies, pitfalls). The agent will gain points by killing enemies, but it will not die instantly from enemy attacks given the agent’s ability to take multiple hits, tracked by the health bar. The agent will gain bonus points by picking up items and lose points by dropping or breaking them (if it is a weapon).  The agent may want to drop items if its inventory is full, so the agent cannot pick up an infinite number of items to gain points infinitely. When the agent exits the dungeon, the score will be finalized by a point multiplier based on remaining health (including extra health from overeating), damage taken, and other factors such as broken weapons. The agent's ultimate goal is to maximize the number of points it can gain given a dungeon.

To visualize the agent’s progress, there will be a secondary window that will track the agent's movement in a top-down view of the dungeon represented as a grid. This will show the entire map and where everything is (the agent, the goal, monsters, item, etc), and each space will be represented by a color representing the reward for landing in that block given the state. It will also output what the agent senses (ie. sight, sound) and what the agent does in response (its output). In our moonshot case, we will increase the complexity of the dungeons by implementing multiple floors to the dungeons with a more variety of weapons (materials made from it), monsters, and traps to learn. We want each floor of the dungeons to be different and increase in difficulty.


# Appointment
-------------
The appointment with Professor Singh took place on: **April 26th, 2019; 10:15-10:30am** at DBH 4204. Thanks to Professor Singh and Reader Yasaman Razeghi for their input in narrowing the scope of our project.

---

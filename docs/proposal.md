# Summary
----------
Our goal in this project is to create an agent that will train to solve a dungeon. These dungeons will be represented by rooms, which will be basic at first, with an exit and entrance. As we make progress we will add complexity such as: 
- enemies that the agent must avoid or destroy
- weapons that the agent can use
- food that the agent can eat
- armor that the agent can wear.
Our dungeons will initially consist of one floor. We eventually want to add a third dimension to the dungeon by creating a ladder up to a higher floor that is bigger and more difficult but offers a greater reward. 

The input will be a dungeon map represented in XML following the Malmo specifications. The output will be a calculated score that is described in more detail in the evaluation plan.

This project’s agent may be applicable to other dungeon crawler games. It could theoretically work with physical agents in the real world: if we set a tiny wheeled robot in a room, this project could be used to help map its environment and move the robot around.

# AI/ML Algorithm
-----------------
We will be using reinforcement learning with variations of either the Markov chains or neuron networks or both

# Evaluation Plan
------------------
The most basic form of evaluation that we will make is the number of steps it takes for the agent to reach the exit of the dungeon. The more steps it takes, the worse of a sore that the agent will get. When training the agent, whenever the agent dies, it will lose a lot of points to teach it that dying is really bad. There will also be items scattered around the dungeons. For each item that agent picks up, it will gain some points and for each item dropped, it will lose the same amount of points. This is to prevent the agent from cheating the system from just picking up item to artificially increase its score. As there is a limit we are placing on the number of items that the agent can carry, this security measure will work. As equipments have durability in the game, when an equipment breaks, the agent will also lose points. The dungeon will also have different enemies spawned. Each enemy killed will give the agent points. Enemy attacks will not instantly kill the agent (given that agent is full hp). When the agent finally exists the dungeon, it will be rewarded with a large sum of points. There will be a point multiplier section based on remaining hp, extra health from overeating, damage taken, and number of equipments broken. The agent's goal is to increase the amount of points it has as much as possible.
To visualize the agent’s progress, there will be a secondary window that will track the agents movement in a top-down view, but in the simple block element form. This will show the entire floor of the dungeon and where everything is (the agent, the goal, monsters, item, etc), and a color scale for the reward of each block. It will also output what the agent senses (sight/sound) and what the agent does in response. In our moonshot case, we will be implementing multiple floors to the dungeons and making the dungeons more complex, with a more variety of weapons (materials made from it), monstars, and traps. Each floor of the dungeons will be different.

---

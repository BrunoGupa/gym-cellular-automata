# Gym Cellular Automata
---

<p align="center">
    <a href="pics/gym_cellular_automata.svg"><img src="pics/gym_cellular_automata.svg"></a>
    <br />
    <br />
    <a href="https://semver.org/"><img src="https://img.shields.io/badge/version-v0.5.1-blue" alt="Semantic Versioning"></a>
    <a href="http://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/license-MIT-red.svg?style=flat" alt="MIT License"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
    <a href="https://gitmoji.dev"><img src="https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg" alt="Gitmoji"></a>
    <br />
    <br />
    <h2 align="center">Cellular Automata Environments for Reinforcement Learning</h2>
</p>
<hr />

_Gym Cellular Automata_ is a collection of _Reinforcement Learning Environments_ (RLEs) that follow the [OpenAI Gym API](https://gym.openai.com/docs).

The available RLEs are based on [Cellular Automata](https://en.wikipedia.org/wiki/Cellular_automaton) (CAs). On them an _Agent_ interacts with a CA, by changing its cell states, in a attempt to drive the emergent properties of its grid to a desired configuration.

## Installation

```bash
git clone https://github.com/elbecerrasoto/gym-cellular-automata
pip install -e gym-cellular-automata
```

## Basic Usage

### Random Policy

:game_die:

```python
import gym

env = gym.make("gym_cellular_automata:ForestFireHelicopter-v0")
obs = env.reset()

total_reward = 0.0
done = False
step = 0
threshold = 12

# Random Policy for at most "threshold" steps
while not done and step < threshold:
    action = env.action_space.sample()  # Your agent goes here!
    obs, reward, done, info = env.step(action)
    total_reward += reward
    step += 1

print(f"Total Steps: {step}")
print(f"Total Reward: {total_reward}")
```

### Available CA envs

```python
import gym_cellular_automata as gymca

# Print available CA envs
print(gymca.REGISTERED_CA_ENVS)
```

## Gallery

### Helicopter ###

![Forest Fire Helicopter](./pics/render_helicopter.svg)

+ [Forest Fire Helicopter](./gym_cellular_automata/forest_fire/helicopter/README.md)

### Bulldozer ###

![Forest Fire Bulldozer](./pics/render_bulldozer.svg)

+ [Forest Fire Bulldozer](./gym_cellular_automata/forest_fire/bulldozer/README.md)

## Documentation

:construction_worker: Documentation is in progress.

+ [Forest Fire Environment Helicopter V0](./gym_cellular_automata/forest_fire/helicopter/README.md)
+ [Forest Fire Environment Bulldozer V1](./gym_cellular_automata/forest_fire/bulldozer/README.md)

## Issues

+ [Known Issues](./issues.md)

## Contributing

:evergreen_tree: :fire:

For contributions with _Forest Fire CA Envs_ read [this!](./gym_cellular_automata/forest_fire/CONTRIBUTING.md)

Contributions to _Gym Cellular Automata_ are always welcome. Feel free to open _pull requests_ at will explaining your proposed change.

As the library is still on early development there is a dire need of everything! (_Utils, Docs, Envs, Tests ..._)

We aim to have a _zoo_ of Cellular Automata Environments. Thus of particular importance is adding more.

This project adheres to the following practices:

+ Workflow: [GitHub flow](https://guides.github.com/introduction/flow/)
+ Style: [Black](https://github.com/psf/black)
+ Test Suite: [Pytest](https://docs.pytest.org/en/stable/index.html)

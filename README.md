```markdown
# 🕷️ Spider-Walk RL Project

> Reinforcement Learning for a simulated multi-leg spider robot built in **PyBullet** and trained using **PPO (Proximal Policy Optimization)**.

---

## 📘 Overview

This project implements a custom reinforcement learning environment where a **spider robot** learns to **walk forward** stably and efficiently.

It includes:
- A fully defined **Gymnasium-compatible PyBullet environment**
- **Parallelized PPO training** with Stable-Baselines3
- **Reward shaping** for locomotion stability, balance, and efficiency
- **VecNormalize** for robust training
- **Human-rendered model testing** and diagnostic reporting

---

## 🧩 Project Structure

```

📂 spider_rl_project/

│

├── spider_env.py          # Custom PyBullet environment (SpiderWalkEnv)

├── Training.py            # Main PPO training script

├── Model_testing.py       # Post-training testing & analysis

├── training_runs/         # Auto-generated folder for saved runs and models

└── README.md              # (this file)

````

---

## 🚀 Setup & Installation

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Training the Model

Run:

```bash
python Training.py
```

This script:

* Creates multiple parallel training environments
* Trains the PPO policy
* Logs progress and model statistics
* Automatically saves:

  * `final_model.zip`
  * `vecnorm_final.pkl`
  * TensorBoard logs in `training_runs/<timestamp>/`

Example training output:

```
✓ Training completed in 29.7m
Final Statistics (last 100 episodes):
  Average Reward: 94.97 ± 66.27
  Average Length: 191 ± 99
```

---

## 🎮 Testing the Trained Model

After training completes, test the model visually:

```bash
python Model_testing.py
```

This script:

* Loads the most recent trained model
* Runs a few evaluation episodes
* Displays the PyBullet GUI
* Prints a summary like:

```
Test Summary:
  Episodes: 3
  Avg Return: 159.56 ± 0.00
  Avg Length: 246 ± 0
```

---

## ⚙️ Environment Details

| Property          | Description                                   |
| ----------------- | --------------------------------------------- |
| Framework         | Gymnasium                                     |
| Physics Engine    | PyBullet                                      |
| Action Space      | Continuous                                    |
| Observation Space | Joint states, orientation, velocity, etc.     |
| Reward Components | Forward velocity, distance, tilt, energy cost |
| Termination       | Timeout or unstable pose                      |

---

## 🧮 Reward Function (Simplified)

```python
reward = (
    2.0 * w_fwd * fwd * self.dt
    + 1.5 * w_dx * dx
    - w_tilt * tilt * self.dt
    - w_vz * abs(vz) * self.dt
    - w_pow * mech_power * self.dt
    + 0.5 * w_cnt * contact_term
)
```

---

## 🧠 Training Tips

* Use headless mode (`render_mode=None`) during training
* Limit CPU usage:

  ```python
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  ```
* Recommended:

  ```python
  N_ENVS = 4
  TOTAL_TIMESTEPS = 3_000_000
  ```
* Add checkpoint callback for auto-saving progress

---

## 🧪 Example Results

| Metric             | Result              |
| ------------------ | ------------------- |
| Training Time      | ~30 minutes         |
| Final Avg Reward   | 94.97 ± 66.27       |
| Avg Episode Length | 191 ± 99            |
| Test Reward        | 159.56              |
| Behavior           | Stable walking gait |

---

## 🏁 Future Improvements

* Add terrain adaptation
* Improve energy efficiency
* Integrate curriculum learning
* Deploy on a real robot

---

## 👨‍💻 Credits

**Project by:** srgtt (O’por)
**Institution:** KMITL Reinforcement Learning Project (3rd Year)

Built using:

* 🧩 Stable-Baselines3
* ⚙️ PyBullet
* 🧠 Gymnasium

---

## 🧾 License

Open-source for educational use.

---

### 🌟 “The spider learns by falling — and standing up again.”

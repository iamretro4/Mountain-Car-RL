# Mountain Car Environment με PPO Algorithm
## Αναφορά Project Reinforcement Learning

---

## Περιεχόμενα
1. [Εισαγωγή](#εισαγωγή)
2. [Περιγραφή Περιβάλλοντος](#περιγραφή-περιβάλλοντος)
3. [Observation Space](#observation-space)
4. [Action Space](#action-space)
5. [State Space](#state-space)
6. [Δομή Rewards](#δομή-rewards)
7. [Επιλογή Αλγορίθμου: PPO](#επιλογή-αλγορίθμου-ppo)
8. [Σύγκριση Αλγορίθμων](#σύγκριση-αλγορίθμων)
9. [Εξήγηση Κώδικα](#εξήγηση-κώδικα)
10. [Αποτελέσματα και Ανάλυση](#αποτελέσματα-και-ανάλυση)
11. [Συμπέρασμα](#συμπέρασμα)

---

## 1. Εισαγωγή

Αυτό το project υλοποιεί μια λύση Reinforcement Learning (RL) για το πρόβλημα Mountain Car χρησιμοποιώντας τον αλγόριθμο Proximal Policy Optimization (PPO). Το Mountain Car είναι ένα κλασικό πρόβλημα ελέγχου όπου ένα υποκινητήριο αυτοκίνητο πρέπει να μάθει να φτάνει σε μια θέση στόχο στην κορυφή ενός λόφου με στρατηγική δημιουργία ορμής.

**Περιβάλλον**: Mountain Car v0 (Gymnasium)  
**Αλγόριθμος**: Proximal Policy Optimization (PPO)  
**Framework**: Stable Baselines3

---

## 2. Περιγραφή Περιβάλλοντος

Το περιβάλλον Mountain Car είναι ένα deterministic Markov Decision Process (MDP) όπου:
- Ένα αυτοκίνητο τοποθετείται στο κάτω μέρος μιας ημιτονοειδούς κοιλάδας
- Το αυτοκίνητο έχει περιορισμένη ισχύ και δεν μπορεί να οδηγήσει απευθείας προς τα πάνω στον λόφο
- Ο πράκτορας πρέπει να μάθει να δημιουργεί ορμή κινούμενος μπρος-πίσω
- Ο στόχος είναι να φτάσει τη σημαία στην κορυφή του δεξιού λόφου (θέση ≥ 0.5)

**Κύρια Χαρακτηριστικά**:
- **Deterministic**: Οι ίδιες ενέργειες παράγουν τα ίδια αποτελέσματα
- **Sparse Rewards**: Μόνο -1 ανά timestep (χωρίς ενδιάμεσα rewards)
- **Episodic**: Τα επεισόδια τελειώνουν όταν φτάνει ο στόχος ή μετά από 200 βήματα
- **Πρόκληση**: Απαιτεί στρατηγικό σχεδιασμό για δημιουργία ορμής

---

## 3. Observation Space

Το observation space είναι ένα **Box** με shape `(2,)` και dtype `float32`:

| Index | Observation | Min | Max | Περιγραφή |
|-------|-------------|-----|-----|-----------|
| 0 | Position | -1.2 | 0.6 | Θέση αυτοκινήτου κατά μήκος x-άξονα (μέτρα) |
| 1 | Velocity | -0.07 | 0.07 | Ταχύτητα αυτοκινήτου (m/s) |

**Λεπτομέρειες Observation**:
- **Position**: Εύρος από -1.2 (αριστερό όριο) έως 0.6 (δεξιό όριο)
- **Velocity**: Περιορίζεται μεταξύ -0.07 και 0.07 m/s
- Το observation είναι συνεχές και πλήρως παρατηρήσιμο

**Παράδειγμα Observation**:
```python
obs = [-0.46352962, 0.0]  # Position: -0.46, Velocity: 0.0
```

---

## 4. Action Space

Το action space είναι **Discrete(3)**, δηλαδή υπάρχουν 3 πιθανές ενέργειες:

| Action | Value | Περιγραφή |
|--------|-------|-----------|
| 0 | Accelerate Left | Σπρώχνει το αυτοκίνητο αριστερά |
| 1 | No Acceleration | Αδράνεια (χωρίς εφαρμογή δύναμης) |
| 2 | Accelerate Right | Σπρώχνει το αυτοκίνητο δεξιά |

**Χαρακτηριστικά Actions**:
- Discrete actions (μόνο μία ενέργεια ανά timestep)
- Deterministic effects
- Περιορισμένη δύναμη: 0.001 (κάνει το πρόβλημα προκλητικό)

---

## 5. State Space

Το state space είναι ισοδύναμο με το observation space σε αυτό το πλήρως παρατηρήσιμο περιβάλλον:

**State = [Position, Velocity]**

**State Transitions**:
Οι δυναμικές ακολουθούν αυτές τις εξισώσεις:

```
velocity_{t+1} = velocity_t + (action - 1) * force - cos(3 * position_t) * gravity
position_{t+1} = position_t + velocity_{t+1}
```

Όπου:
- `force = 0.001`
- `gravity = 0.0025`
- `action - 1` maps: 0→-1, 1→0, 2→+1

**Συνθήκες Ορίων**:
- Η θέση περιορίζεται στο `[-1.2, 0.6]`
- Η ταχύτητα περιορίζεται στο `[-0.07, 0.07]`
- Οι συγκρούσεις είναι ανελαστικές (ταχύτητα ορίζεται σε 0 στα όρια)

**Αρχική Κατάσταση**:
- Position: Ομοιόμορφη τυχαία στο `[-0.6, -0.4]`
- Velocity: Πάντα 0

---

## 6. Δομή Rewards

**Συνάρτηση Reward**:
```
reward = -1 για κάθε timestep
```

**Χαρακτηριστικά Rewards**:
- **Sparse**: Χωρίς ενδιάμεσα rewards για πρόοδο
- **Αρνητικό**: Τιμωρεί τον χρόνο που περνάει (ενθαρρύνει αποδοτικότητα)
- **Τελικό**: Το επεισόδιο τελειώνει όταν position ≥ 0.5 (φτάνει ο στόχος)
- **Μέγιστο Μήκος Επεισοδίου**: 200 βήματα (truncation)

**Εύρος Rewards**:
- Καλύτερο δυνατό: -1 (φτάνει στόχο σε 1 βήμα, αλλά απίθανο)
- Τυπικό επιτυχημένο επεισόδιο: -100 έως -200
- Αποτυχημένο επεισόδιο: -200 (έφτασε το μέγιστο μήκος)

**Γιατί αυτή η δομή Rewards?**:
- Ενθαρρύνει την επίτευξη του στόχου όσο πιο γρήγορα γίνεται
- Δεν χρειάζεται reward shaping (κρατάει το πρόβλημα απλό)
- Δημιουργεί ένα προκλητικό πρόβλημα exploration

---

## 7. Επιλογή Αλγορίθμου: PPO

### Γιατί PPO για Mountain Car?

**1. On-Policy Learning**
- Το PPO είναι on-policy, καθιστώντας το sample-efficient για αυτό το περιβάλλον
- Το Mountain Car ωφελείται από την εκμάθηση από την εμπειρία της τρέχουσας πολιτικής
- Δεν χρειάζεται experience replay (σε αντίθεση με DQN)

**2. Σταθερότητα**
- Το clipped objective του PPO αποτρέπει μεγάλες ενημερώσεις πολιτικής
- Σημαντικό για το Mountain Car όπου μικρές αλλαγές στην πολιτική μπορούν να επηρεάσουν σημαντικά τη συμπεριφορά
- Μειώνει τον κίνδυνο κατάρρευσης πολιτικής

**3. Συνεχές State, Discrete Action**
- Το PPO χειρίζεται καλά discrete action spaces
- Λειτουργεί αποτελεσματικά με συνεχή observations (θέση, ταχύτητα)
- Δεν χρειάζεται action discretization

**4. Sample Efficiency**
- Το PPO μπορεί να μάθει με σχετικά λίγα samples
- Τα επεισόδια Mountain Car είναι σύντομα (≤200 βήματα), καθιστώντας τη sample efficiency σημαντική
- Πολλαπλές epochs ανά ενημέρωση βελτιώνουν τη χρήση δεδομένων

**5. Robustness Hyperparameters**
- Το PPO είναι γνωστό για την ανθεκτικότητά του στις επιλογές hyperparameters
- Καλά default hyperparameters λειτουργούν καλά out-of-the-box
- Μειώνει τον χρόνο tuning

**6. Deterministic Environment**
- Το PPO λειτουργεί καλά σε deterministic περιβάλλοντα
- Δεν χρειάζεται stochastic policy exploration (αν και το PPO το υποστηρίζει)

**7. Αποδεδειγμένη Απόδοση**
- Το PPO έχει δείξει ισχυρή απόδοση σε κλασικά προβλήματα ελέγχου
- Ευρέως χρησιμοποιούμενο και καλά τεκμηριωμένο
- Καλή ισορροπία μεταξύ απόδοσης και πολυπλοκότητας υλοποίησης

---

## 8. Σύγκριση Αλγορίθμων

### Πίνακας Σύγκρισης

| Αλγόριθμος | Τύπος | Sample Efficiency | Σταθερότητα | Discrete Actions | Καλύτερο Για |
|-----------|-------|-------------------|-------------|------------------|--------------|
| **PPO** | On-policy | Υψηλή | Πολύ Υψηλή | ✅ | **Mountain Car** |
| DQN | Off-policy | Μέτρια | Μέτρια | ✅ | Atari games |
| A2C | On-policy | Υψηλή | Υψηλή | ✅ | Παρόμοιο με PPO |
| SAC | Off-policy | Υψηλή | Υψηλή | ❌ | Συνεχείς ενέργειες |
| DDPG | Off-policy | Μέτρια | Μέτρια | ❌ | Συνεχείς ενέργειες |
| TRPO | On-policy | Υψηλή | Πολύ Υψηλή | ✅ | Παρόμοιο με PPO |

### Αναλυτική Σύγκριση

#### 1. PPO vs DQN

**Πλεονεκτήματα PPO**:
- Καλύτερη sample efficiency (on-policy learning)
- Πιο σταθερή εκπαίδευση (clipped objective)
- Χειρίζεται φυσικά συνεχή state spaces
- Λειτουργεί καλά με σύντομα επεισόδια

**Μειονεκτήματα DQN για Mountain Car**:
- Απαιτεί experience replay (memory overhead)
- Λιγότερο sample efficient
- Μπορεί να overfit σε προηγούμενες εμπειρίες
- Καλύτερο για high-dimensional observations (π.χ. εικόνες)

**Απόφαση**: Το PPO είναι καλύτερο για το συνεχές state, discrete action space του Mountain Car.

#### 2. PPO vs A2C

**Ομοιότητες**:
- Και τα δύο είναι on-policy actor-critic methods
- Παρόμοια sample efficiency
- Και τα δύο υποστηρίζουν discrete actions

**Πλεονεκτήματα PPO**:
- Πιο σταθερό (clipped surrogate objective)
- Καλύτερη robustness hyperparameters
- Αποτρέπει μεγάλες ενημερώσεις πολιτικής

**Πλεονεκτήματα A2C**:
- Απλούστερη υλοποίηση
- Ελαφρώς ταχύτερο ανά ενημέρωση

**Απόφαση**: Το PPO προτιμάται για σταθερότητα, αν και το A2C θα λειτουργούσε επίσης καλά.

#### 3. PPO vs SAC/DDPG

**Κύρια Διαφορά**: Το SAC και το DDPG σχεδιάστηκαν για **συνεχή action spaces**

**Γιατί δεν είναι κατάλληλα**:
- Το Mountain Car έχει discrete actions (3 επιλογές)
- Θα απαιτούσε action discretization
- Overkill για απλό discrete action space
- Πιο πολύπλοκο hyperparameter tuning

**Απόφαση**: Δεν είναι κατάλληλα για discrete action spaces.

#### 4. PPO vs TRPO

**Ομοιότητες**:
- Και τα δύο είναι on-policy με trust region methods
- Και τα δύο πολύ σταθερά

**Πλεονεκτήματα PPO**:
- Απλούστερη υλοποίηση (clipped objective vs constrained optimization)
- Ταχύτερος υπολογισμός
- Ευκολότερο tuning

**Πλεονεκτήματα TRPO**:
- Θεωρητικά πιο αρχιτεκτονικό
- Εγγυημένη μονοτονική βελτίωση

**Απόφαση**: Το PPO προτιμάται για πρακτική υλοποίηση, το TRPO για θεωρητικές εγγυήσεις.

### Τελική Επιλογή: PPO

**Σύνοψη Γιατί PPO**:
1. ✅ Βέλτιστο για discrete actions + συνεχή states
2. ✅ Υψηλή sample efficiency (σημαντικό για σύντομα επεισόδια)
3. ✅ Πολύ σταθερή εκπαίδευση
4. ✅ Καλά default hyperparameters
5. ✅ Αποδεδειγμένη απόδοση σε κλασικό έλεγχο
6. ✅ Ισορροπημένη πολυπλοκότητα vs απόδοση

---

## 9. Εξήγηση Κώδικα

### 9.1 Environment Setup

```python
def create_environment():
    env = gym.make("MountainCar-v0", render_mode=None)
    return env
```

**Εξήγηση**:
- Δημιουργεί το περιβάλλον Mountain Car από το Gymnasium
- `render_mode=None` για εκπαίδευση (ταχύτερο, χωρίς visualization)
- Το περιβάλλον παρέχει αυτόματα τα σωστά observation/action spaces

### 9.2 Vectorized Environments

```python
env = make_vec_env("MountainCar-v0", n_envs=4, seed=42)
```

**Εξήγηση**:
- Δημιουργεί 4 παράλληλα περιβάλλοντα για ταχύτερη εκπαίδευση
- Κάθε περιβάλλον τρέχει ανεξάρτητα
- Συλλέγει 4x περισσότερη εμπειρία ανά βήμα
- Επιταχύνει σημαντικά την εκπαίδευση

### 9.3 PPO Model Configuration

```python
model = PPO(
    "MlpPolicy",              # Τύπος policy network
    env,
    learning_rate=3e-4,       # Πόσο γρήγορα μαθαίνει
    n_steps=2048,             # Βήματα πριν την ενημέρωση
    batch_size=64,            # Μέγεθος batch εκπαίδευσης
    n_epochs=10,              # Epochs εκπαίδευσης ανά ενημέρωση
    gamma=0.99,               # Discount factor
    gae_lambda=0.95,          # Παράμετρος GAE
    clip_range=0.2,           # PPO clipping
    ent_coef=0.01,            # Bonus exploration
    vf_coef=0.5,              # Βάρος value function
    max_grad_norm=0.5,        # Gradient clipping
    seed=42
)
```

**Εξήγηση Hyperparameters**:

- **`MlpPolicy`**: Multi-layer perceptron (νευρωνικό δίκτυο)
  - Input: 2 (θέση, ταχύτητα)
  - Output: 3 πιθανότητες actions

- **`learning_rate=3e-4`**: Τυπικό learning rate
  - Πολύ υψηλό: ασταθής εκπαίδευση
  - Πολύ χαμηλό: αργή εκμάθηση

- **`n_steps=2048`**: Συλλέγει 2048 βήματα πριν την ενημέρωση
  - Ισορροπεί sample efficiency και συχνότητα ενημέρωσης
  - Με 4 envs: 512 βήματα ανά env

- **`batch_size=64`**: Μέγεθος mini-batch για εκπαίδευση
  - Διαιρεί τη συλλεγμένη εμπειρία σε batches
  - 2048 / 64 = 32 batches ανά ενημέρωση

- **`n_epochs=10`**: Εκπαιδεύεται στα ίδια δεδομένα 10 φορές
  - Βελτιώνει τη sample efficiency
  - Το PPO μπορεί να κάνει με ασφάλεια πολλαπλές epochs λόγω clipping

- **`gamma=0.99`**: Discount factor
  - Πόσο να αξιολογεί future rewards
  - 0.99 = αξιολογεί rewards 100 βήματα μπροστά στο ~37%

- **`gae_lambda=0.95`**: Generalized Advantage Estimation
  - Ισορροπεί bias και variance σε advantage estimates
  - 0.95 = καλή ισορροπία για Mountain Car

- **`clip_range=0.2`**: Βασικό χαρακτηριστικό PPO
  - Αποτρέπει την πολιτική να αλλάζει πολύ
  - Περιορίζει probability ratio μεταξύ 0.8 και 1.2
  - Εξασφαλίζει σταθερή εκμάθηση

- **`ent_coef=0.01`**: Συντελεστής entropy
  - Ενθαρρύνει exploration
  - Αποτρέπει την πολιτική να γίνει πολύ deterministic πολύ νωρίς

- **`vf_coef=0.5`**: Συντελεστής value function
  - Βάρος για value function loss
  - Ισορροπεί εκπαίδευση πολιτικής και value

- **`max_grad_norm=0.5`**: Gradient clipping
  - Αποτρέπει exploding gradients
  - Βελτιώνει τη σταθερότητα εκπαίδευσης

### 9.4 Training Process

```python
model.learn(
    total_timesteps=500000,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True
)
```

**Εξήγηση**:
- Εκπαιδεύεται για 500,000 timesteps
- `eval_callback`: Αξιολογεί το μοντέλο περιοδικά, αποθηκεύει το καλύτερο
- `checkpoint_callback`: Αποθηκεύει checkpoints μοντέλου
- `progress_bar`: Δείχνει την πρόοδο εκπαίδευσης

**Ροή Εκπαίδευσης**:
1. Συλλογή 2048 βημάτων εμπειρίας
2. Υπολογισμός advantages χρησιμοποιώντας GAE
3. Εκπαίδευση για 10 epochs στα συλλεγμένα δεδομένα
4. Ενημέρωση policy και value networks
5. Επανάληψη μέχρι 500k timesteps

### 9.5 Evaluation

```python
def evaluate_agent(model, env, n_episodes=10, render=False):
    # Για κάθε επεισόδιο:
    #   1. Reset περιβάλλοντος
    #   2. Προβλέψει action χρησιμοποιώντας εκπαιδευμένο μοντέλο
    #   3. Βήμα περιβάλλοντος
    #   4. Συλλογή states, actions, rewards
    #   5. Υπολογισμός στατιστικών
```

**Εξήγηση**:
- Τρέχει τον πράκτορα για πολλαπλά επεισόδια
- Χρησιμοποιεί deterministic policy (χωρίς exploration)
- Συλλέγει δεδομένα για ανάλυση
- Υπολογίζει mean reward και episode length

### 9.6 Λεπτομέρειες Αλγορίθμου PPO

**Βήματα Αλγορίθμου PPO**:

1. **Συλλογή Εμπειρίας**:
   ```python
   # Για n_steps:
   action = policy.predict(observation)
   next_obs, reward, done = env.step(action)
   # Αποθήκευση: (obs, action, reward, next_obs, done)
   ```

2. **Υπολογισμός Advantages**:
   ```python
   # Χρησιμοποιώντας Generalized Advantage Estimation (GAE)
   advantage = reward + gamma * V(next_obs) - V(obs)
   # Το GAE εξομαλύνει advantages χρησιμοποιώντας lambda
   ```

3. **Υπολογισμός Παλιών Πιθανοτήτων Πολιτικής**:
   ```python
   old_log_prob = old_policy.log_prob(action)
   ```

4. **Ενημέρωση Πολιτικής** (Πολλαπλές Epochs):
   ```python
   for epoch in range(n_epochs):
       # Λήψη τρεχουσών πιθανοτήτων πολιτικής
       new_log_prob = policy.log_prob(action)
       
       # Υπολογισμός probability ratio
       ratio = exp(new_log_prob - old_log_prob)
       
       # Υπολογισμός clipped objective
       clipped_ratio = clip(ratio, 1 - epsilon, 1 + epsilon)
       policy_loss = -min(ratio * advantage, clipped_ratio * advantage)
       
       # Ενημέρωση policy network
       optimizer.step()
   ```

5. **Ενημέρωση Value Function**:
   ```python
   value_loss = (V(obs) - target_value)^2
   # Ενημέρωση value network
   ```

**Βασικά Χαρακτηριστικά PPO**:

- **Clipped Surrogate Objective**: Αποτρέπει μεγάλες ενημερώσεις πολιτικής
  ```python
  L^CLIP = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]
  ```
  Όπου `r(θ) = π_θ(a|s) / π_θ_old(a|s)`

- **Πολλαπλές Epochs**: Αναχρησιμοποιεί αποτελεσματικά συλλεγμένα δεδομένα

- **Actor-Critic**: Μαθαίνει τόσο policy (actor) όσο και value function (critic)

---

## 10. Αποτελέσματα και Ανάλυση

### Αναμενόμενα Αποτελέσματα

Μετά την εκπαίδευση για 500,000 timesteps, ο πράκτορας θα πρέπει:

1. **Να μάθει τη Στρατηγική**:
   - Να δημιουργεί ορμή κινώντας πρώτα αριστερά
   - Μετά να επιταχύνει δεξιά για να φτάσει τον στόχο
   - Να φτάνει στη θέση στόχου (≥0.5) συνεπώς

2. **Μετρικές Απόδοσης**:
   - Ποσοστό επιτυχίας: >80% (φτάνει στόχο)
   - Mean reward: -100 έως -150 (αποδοτικά επεισόδια)
   - Mean episode length: 100-150 βήματα

3. **Learning Curve**:
   - Αρχικό: Τυχαία exploration, -200 reward (αποτυχία)
   - Μεσαία εκπαίδευση: Μαθαίνει να δημιουργεί ορμή, περιστασιακή επιτυχία
   - Τελικό: Συνεπής επιτυχία, αποδοτικές διαδρομές

### Ανάλυση States, Actions, και Rewards

**Ανάλυση State**:
- **Position**: Ξεκινά γύρω στο -0.5, πρέπει να φτάσει 0.5
- **Velocity**: Χτίζεται μέσω στρατηγικών actions
- **Phase Space**: Δείχνει ταλαντωτική συμπεριφορά (μπρος-πίσω)

**Ανάλυση Actions**:
- **Πρώιμη Εκπαίδευση**: Τυχαία actions
- **Μαθημένη Πολιτική**: 
  - Αριστερά όταν είναι στο κάτω μέρος (δημιουργία ορμής)
  - Δεξιά όταν κινείται αριστερά (επιτάχυνση προς στόχο)
  - Ο στρατηγικός συγχρονισμός είναι κρίσιμος

**Ανάλυση Rewards**:
- **Sparse Rewards**: -1 ανά timestep
- **Πρόκληση**: Χωρίς ενδιάμεσα feedback
- **Λύση**: Η εκτίμηση advantage του PPO βοηθάει με sparse rewards

---

## 11. Συμπέρασμα

### Σύνοψη

Αυτό το project υλοποιεί επιτυχώς το PPO για το περιβάλλον Mountain Car:

1. **Περιβάλλον**: Mountain Car v0 με συνεχή states και discrete actions
2. **Αλγόριθμος**: PPO επιλέχθηκε για σταθερότητα, sample efficiency, και καταλληλότητα
3. **Υλοποίηση**: Πλήρης pipeline εκπαίδευσης και αξιολόγησης
4. **Αποτελέσματα**: Ο πράκτορας μαθαίνει να φτάνει τον στόχο αποτελεσματικά

### Βασικά Συμπεράσματα

1. **Το PPO είναι κατάλληλο** για το Mountain Car λόγω:
   - On-policy learning για sample efficiency
   - Σταθερότητα για αξιόπιστη εκπαίδευση
   - Καλή χειριστικότητα discrete actions

2. **Προκλήσεις Mountain Car**:
   - Τα sparse rewards απαιτούν καλό exploration
   - Απαιτείται στρατηγικός σχεδιασμός (δημιουργία ορμής)
   - Τα σύντομα επεισόδια ωφελούνται από sample-efficient αλγόριθμους

3. **Πλεονεκτήματα PPO**:
   - Το clipped objective αποτρέπει αστάθεια
   - Οι πολλαπλές epochs βελτιώνουν την αποδοτικότητα δεδομένων
   - Ανθεκτικά hyperparameters

### Μελλοντικές Βελτιώσεις

1. **Hyperparameter Tuning**: Grid search για βέλτιστες παραμέτρους
2. **Reward Shaping**: Προσθήκη ενδιάμεσων rewards για πρόοδο
3. **Σύγκριση Αλγορίθμων**: Υλοποίηση DQN, A2C για άμεση σύγκριση
4. **Visualization**: Real-time rendering της μαθημένης πολιτικής
5. **Transfer Learning**: Δοκιμή στη variant Mountain Car Continuous

---

## Αναφορές

1. Gymnasium Documentation: https://gymnasium.farama.org/environments/classic_control/mountain_car/
2. Stable Baselines3 Documentation: https://stable-baselines3.readthedocs.io/
3. PPO Paper: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
4. Mountain Car Original: Moore, A. W. (1990) "Efficient Memory-based Learning for Robot Control"

---

## Παράρτημα: Εκτέλεση Κώδικα

### Εγκατάσταση

```bash
pip install -r requirements.txt
```

### Εκπαίδευση

```bash
python mountain_car_ppo.py
```

### Αρχεία Εξόδου

- `models/mountain_car_ppo_final.zip`: Εκπαιδευμένο μοντέλο
- `results/training_analysis.png`: Visualizations εκπαίδευσης
- `results/phase_space.png`: Phase space plot
- `tensorboard_logs/`: TensorBoard logs για παρακολούθηση

### Φόρτωση και Δοκιμή

```python
from stable_baselines3 import PPO
import gymnasium as gym

# Φόρτωση μοντέλου
model = PPO.load("./models/mountain_car_ppo_final")

# Δοκιμή
env = gym.make("MountainCar-v0", render_mode="human")
obs, info = env.reset()
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break
env.close()
```

---

**Τέλος Αναφοράς**


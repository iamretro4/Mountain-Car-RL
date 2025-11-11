# Mountain Car RL Algorithm - Υλοποίηση PPO

Αυτό το project υλοποιεί έναν αλγόριθμο Proximal Policy Optimization (PPO) για την επίλυση του περιβάλλοντος Mountain Car από το Gymnasium.

## Δομή Project

```
Mountain Car RL Algorithm/
├── mountain_car_ppo.py    # Κύριο script εκπαίδευσης
├── requirements.txt       # Python dependencies
├── REPORT.md             # Αναλυτική αναφορά project
└── README.md             # Αυτό το αρχείο
```

## Γρήγορη Έναρξη

### 1. Εγκατάσταση Dependencies

```bash
pip install -r requirements.txt
```

### 2. Εκτέλεση Εκπαίδευσης

```bash
python mountain_car_ppo.py
```

Αυτό θα:
- Εκπαιδεύσει έναν πράκτορα PPO για 500,000 timesteps
- Αποθηκεύσει το εκπαιδευμένο μοντέλο στο `./models/`
- Δημιουργήσει αποτελέσματα αξιολόγησης και visualizations στο `./results/`

### 3. Προβολή Αποτελεσμάτων

Μετά την εκπαίδευση, ελέγξτε:
- `./models/mountain_car_ppo_final.zip` - Τελικό εκπαιδευμένο μοντέλο
- `./results/training_analysis.png` - Visualizations εκπαίδευσης
- `./results/phase_space.png` - Phase space plot
- `./tensorboard_logs/` - TensorBoard logs (προβολή με `tensorboard --logdir ./tensorboard_logs/`)

## Λεπτομέρειες Περιβάλλοντος

- **Περιβάλλον**: Mountain Car v0 (Gymnasium)
- **Observation Space**: Box(2,) - [θέση, ταχύτητα]
- **Action Space**: Discrete(3) - [αριστερά, χωρίς επιτάχυνση, δεξιά]
- **Reward**: -1 ανά timestep (sparse rewards)
- **Στόχος**: Φτάσει στη θέση ≥ 0.5

## Αλγόριθμος

- **Αλγόριθμος**: Proximal Policy Optimization (PPO)
- **Framework**: Stable Baselines3
- **Policy**: Multi-layer Perceptron (MlpPolicy)

## Τεκμηρίωση

Δείτε το `REPORT.md` για:
- Αναλυτική περιγραφή περιβάλλοντος
- Αιτιολόγηση επιλογής αλγορίθμου
- Εξηγήσεις κώδικα
- Σύγκριση με άλλους RL αλγόριθμους
- Ανάλυση αποτελεσμάτων

## Απαιτήσεις

- Python 3.8+
- PyTorch
- Gymnasium
- Stable Baselines3
- NumPy
- Matplotlib

## Σημειώσεις

- Η εκπαίδευση μπορεί να πάρει 30-60 λεπτά ανάλογα με το hardware σας
- Το μοντέλο θα αποθηκεύεται περιοδικά κατά τη διάρκεια της εκπαίδευσης
- Χρησιμοποιήστε το TensorBoard για παρακολούθηση της προόδου εκπαίδευσης σε πραγματικό χρόνο


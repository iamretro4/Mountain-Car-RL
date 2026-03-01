# Mountain Car RL Algorithm - Υλοποίηση PPO

Αυτό το project υλοποιεί έναν αλγόριθμο Proximal Policy Optimization (PPO) για την επίλυση του περιβάλλοντος Mountain Car από το Gymnasium.

## 🚗 Επισκόπηση Project

Αυτό το project εκπαιδεύει έναν AI πράκτορα να λύσει το κλασικό πρόβλημα Mountain Car, όπου ένα υποκινητήριο αυτοκίνητο πρέπει να μάθει να δημιουργεί ορμή και να φτάσει σε μια θέση στόχο στην κορυφή ενός λόφου.

## ✨ Χαρακτηριστικά

- **Υλοποίηση Αλγορίθμου PPO** χρησιμοποιώντας Stable Baselines3
- **Reward Shaping** - Ενδιάμεσα σήματα ανταμοιβής για αποτελεσματική εκμάθηση
- **Real-time Training Visualization** - Παρακολουθήστε το αυτοκίνητο να μαθαίνει σε πραγματικό χρόνο
- **Διαδραστικά Dashboards** - Περιεκτικά HTML visualizations
- **Πλήρης Training Pipeline** - Από εκπαίδευση έως αξιολόγηση
- **Ενσωμάτωση TensorBoard** - Παρακολούθηση προόδου εκπαίδευσης

## 📋 Απαιτήσεις

- Python 3.8+
- Δείτε το `requirements.txt` για όλες τις dependencies

## 🚀 Γρήγορη Έναρξη

### 1. Εγκατάσταση Dependencies

```bash
pip install -r requirements.txt
```

### 2. Εκπαίδευση με Visualization

```bash
python train_with_visualization.py
```

Αυτό θα:
- Εκπαιδεύσει έναν πράκτορα PPO για 500,000 timesteps
- Δείξει το αυτοκίνητο να μαθαίνει κάθε 50 επεισόδια
- Αποθηκεύσει metrics αυτόματα
- Δημιουργήσει visualizations στο τέλος

### 3. Προβολή Αποτελεσμάτων

```bash
# Δημιουργία διαδραστικών dashboards
python visualize_training.py

# Άνοιγμα στον browser
start .\visualizations\main_dashboard.html

# Παρακολούθηση εκπαιδευμένου πράκτορα
python view_agent.py
```

## 📁 Δομή Project

```
Mountain Car RL Algorithm/
├── mountain_car_ppo.py          # Κύριο training script
├── train_with_visualization.py  # Εκπαίδευση με real-time visualization
├── visualize_training.py        # Generator διαδραστικών dashboards
├── view_agent.py                # Παρακολούθηση εκπαιδευμένου πράκτορα
├── check_training_status.py     # Εργαλείο διαγνωστικής
├── extract_tensorboard_metrics.py # Εξαγωγή metrics από TensorBoard
├── requirements.txt             # Python dependencies
├── README.md                    # Αυτό το αρχείο
├── REPORT.md                    # Αναλυτική αναφορά project
├── VISUALIZATION_GUIDE.md       # Οδηγός visualizations
├── ΟΔΗΓΙΕΣ_ΕΚΤΕΛΕΣΗΣ_EL.md     # Οδηγίες εκτέλεσης
├── models/                      # Εκπαιδευμένα μοντέλα
├── results/                     # Αποτελέσματα εκπαίδευσης και plots
├── visualizations/              # Διαδραστικά HTML dashboards
├── training_logs/              # Training metrics
└── tensorboard_logs/            # TensorBoard logs
```

## 🎯 Λεπτομέρειες Περιβάλλοντος

- **Περιβάλλον**: Mountain Car v0 (Gymnasium)
- **Observation Space**: Box(2,) - [θέση, ταχύτητα]
- **Action Space**: Discrete(3) - [αριστερά, χωρίς επιτάχυνση, δεξιά]
- **Reward (default)**: -1 ανά timestep (sparse rewards)
- **Στόχος**: Φτάσει στη θέση ≥ 0.5

## 🧠 Αλγόριθμος

- **Αλγόριθμος**: Proximal Policy Optimization (PPO)
- **Framework**: Stable Baselines3
- **Policy**: Multi-layer Perceptron (MlpPolicy)
- **Reward Shaping**: Wrapper (`ShapedMountainCar`) που προσθέτει ενδιάμεσα σήματα μάθησης

### Reward Shaping (Διαμόρφωση Ανταμοιβής)

Το default reward του Mountain Car (-1 ανά βήμα) είναι πολύ αραιό (sparse) -- ο πράκτορας δεν λαμβάνει κανένα σήμα προόδου μέχρι να φτάσει τον στόχο. Με τυχαία εξερεύνηση, αυτό σχεδόν ποτέ δεν συμβαίνει, οπότε ο πράκτορας δεν μαθαίνει τίποτα.

Η λύση είναι ο wrapper `ShapedMountainCar` που προσθέτει τρία μικρά ενδιάμεσα σήματα:

1. **Μπόνους νέου ύψους** -- όταν το αυτοκίνητο φτάνει σε θέση ψηλότερη από οποιαδήποτε προηγούμενη στο ίδιο επεισόδιο
2. **Μπόνους θέσης** -- μικρή συνεχής ανταμοιβή βάσει του πόσο ψηλά βρίσκεται το αυτοκίνητο
3. **Μπόνους ταχύτητας** -- ανταμοιβή για κινητική ενέργεια (ταχύτητα σε οποιαδήποτε κατεύθυνση)

Η αξιολόγηση γίνεται στο **πραγματικό περιβάλλον χωρίς shaping**, ώστε τα αποτελέσματα να είναι γνήσια.

## 📊 Αποτελέσματα

Μετά την εκπαίδευση 500,000 timesteps:
- **Ποσοστό Επιτυχίας**: **100%**
- **Mean Reward**: -117.10 ± 2.59
- **Mean Episode Length**: ~117 βήματα

## 📚 Τεκμηρίωση

- **REPORT.md** - Αναλυτική αναφορά με περιγραφή περιβάλλοντος, επιλογή αλγορίθμου, εξηγήσεις κώδικα, σύγκριση αλγορίθμων και ανάλυση αποτελεσμάτων
- **VISUALIZATION_GUIDE.md** - Οδηγός για τα διαδραστικά visualization tools
- **ΟΔΗΓΙΕΣ_ΕΚΤΕΛΕΣΗΣ_EL.md** - Πλήρεις οδηγίες εκτέλεσης

## 🛠️ Παραδείγματα Χρήσης

### Εκπαίδευση χωρίς visualization (πιο γρήγορο)
```bash
python mountain_car_ppo.py
```

### Εκπαίδευση με πιο συχνό visualization
```bash
python train_with_visualization.py --freq 25
```

### Έλεγχος κατάστασης εκπαίδευσης
```bash
python check_training_status.py
```

### Εξαγωγή metrics από TensorBoard
```bash
python extract_tensorboard_metrics.py
```

## 📈 Παρακολούθηση

- **TensorBoard**: `tensorboard --logdir ./tensorboard_logs/`
- **Διαδραστικά Dashboards**: Άνοιγμα `./visualizations/main_dashboard.html`
- **Training Metrics**: Αποθηκεύονται αυτόματα στο `./training_logs/`

## 📝 Σημειώσεις

- **Χρόνος Εκπαίδευσης**: ~2 λεπτά (500k timesteps, ανάλογα με το hardware)
- **Real-time Visualization**: Μπορεί να επιβραδύνει λίγο την εκπαίδευση
- **Metrics**: Αποθηκεύονται αυτόματα κάθε 5 επεισόδια
- **Checkpoints**: Αποθηκεύονται κάθε 50,000 timesteps

## 🔗 Σύνδεσμοι

- [Gymnasium Documentation](https://gymnasium.farama.org/environments/classic_control/mountain_car/)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)

---

**Απολαύστε την εκπαίδευση του Mountain Car πράκτορα! 🚗⛰️**

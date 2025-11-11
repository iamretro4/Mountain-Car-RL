# Οδηγίες Εκτέλεσης - Mountain Car RL Algorithm

## 📋 Σειρά Εκτέλεσης για Πλήρη Project

### Βήμα 1: Εγκατάσταση Dependencies

```bash
cd "C:\Users\anton\OneDrive\Documents\Mountain Car RL Algorithm"
pip install -r requirements.txt
```

### Βήμα 2: Εκπαίδευση με Visualization (Προτεινόμενο)

**Επιλογή Α: Εκπαίδευση με Real-time Visualization**
```bash
python train_with_visualization.py
```

Αυτό θα:
- Εκπαιδεύσει τον πράκτορα PPO
- Δείξει το αυτοκίνητο να παίζει κάθε 50 επεισόδια (προσαρμόσιμο)
- Αποθηκεύσει metrics αυτόματα
- Δημιουργήσει visualizations στο τέλος

**Επιλογές:**
```bash
# Δείξε πρόοδο κάθε 25 επεισόδια (πιο συχνά)
python train_with_visualization.py --freq 25

# Χωρίς real-time visualization (πιο γρήγορο)
python train_with_visualization.py --no-visual
```

**Επιλογή Β: Κανονική Εκπαίδευση**
```bash
python mountain_car_ppo.py
```

### Βήμα 3: Έλεγχος Κατάστασης (Προαιρετικό)

```bash
python check_training_status.py
```

Αυτό θα σας δείξει:
- Αν υπάρχουν training metrics
- Αν υπάρχουν μοντέλα
- Αν υπάρχουν TensorBoard logs

### Βήμα 4: Εξαγωγή Metrics από TensorBoard (Αν χρειάζεται)

Αν η εκπαίδευση ολοκληρώθηκε αλλά δεν υπάρχουν metrics:

```bash
python extract_tensorboard_metrics.py
```

### Βήμα 5: Δημιουργία Interactive Dashboards

```bash
python visualize_training.py
```

Αυτό δημιουργεί:
- `./visualizations/training_dashboard.html` - Πρόοδος εκπαίδευσης
- `./visualizations/hyperparameters.html` - Hyperparameters
- `./visualizations/agent_performance.html` - Απόδοση πράκτορα
- `./visualizations/summary_dashboard.html` - Περίληψη
- `./visualizations/main_dashboard.html` - Κύριο dashboard

### Βήμα 6: Προβολή Dashboards

Ανοίξτε στον browser:
```
.\visualizations\main_dashboard.html
```

Ή απλά:
```bash
start .\visualizations\main_dashboard.html
```

### Βήμα 7: Προβολή Πράκτορα σε Δράση

```bash
# Προβολή 5 επεισοδίων (προεπιλογή)
python view_agent.py

# Προβολή 10 επεισοδίων
python view_agent.py ./models/mountain_car_ppo_final.zip 10

# Ταχύτερη αναπαραγωγή (0.02 δευτερόλεπτα ανά βήμα)
python view_agent.py ./models/mountain_car_ppo_final.zip 5 0.02
```

## 🎬 Γρήγορη Εκτέλεση (Όλα σε Μία)

```bash
# 1. Εγκατάσταση
pip install -r requirements.txt

# 2. Εκπαίδευση με visualization
python train_with_visualization.py

# 3. Δημιουργία dashboards (αυτόματα στο τέλος, ή χειροκίνητα)
python visualize_training.py

# 4. Προβολή dashboards
start .\visualizations\main_dashboard.html

# 5. Προβολή πράκτορα
python view_agent.py
```

## 📊 Real-time Visualization Κατά τη Διάρκεια Εκπαίδευσης

Το `train_with_visualization.py` δείχνει:
- **Κάθε 50 επεισόδια**: Ανοίγει παράθυρο και δείχνει το αυτοκίνητο να παίζει
- **Real-time learning**: Βλέπετε πώς βελτιώνεται ο πράκτορας
- **Progress tracking**: Εμφανίζει max position, rewards, success/failure

**Παράδειγμα Output:**
```
============================================================
Episode 50 - Visualizing Agent Progress
============================================================
Result: [FAILED]
Steps: 200, Max Position: -0.2341, Reward: -200.00
============================================================

============================================================
Episode 100 - Visualizing Agent Progress
============================================================
Result: [SUCCESS]
Steps: 145, Max Position: 0.5234, Reward: -145.00
============================================================
```

## 🔄 Workflow για Συνεχή Ανάπτυξη

1. **Εκπαίδευση**: `python train_with_visualization.py`
2. **Έλεγχος**: `python check_training_status.py`
3. **Visualization**: `python visualize_training.py`
4. **Αξιολόγηση**: `python view_agent.py`
5. **Βελτίωση**: Τροποποίηση hyperparameters και επανάληψη

## ⚙️ Προχωρημένες Επιλογές

### Εκπαίδευση με Custom Hyperparameters

Επεξεργαστείτε το `mountain_car_ppo.py` και αλλάξτε:
- `learning_rate`
- `n_steps`
- `batch_size`
- κ.λπ.

### Παρακολούθηση με TensorBoard

```bash
tensorboard --logdir ./tensorboard_logs/
```

Ανοίγει στον browser στο `http://localhost:6006`

### Εξαγωγή Metrics από Προηγούμενη Εκπαίδευση

Αν έχετε TensorBoard logs αλλά όχι JSON metrics:

```bash
python extract_tensorboard_metrics.py
python visualize_training.py
```

## 📝 Σημειώσεις

- **Χρόνος Εκπαίδευσης**: 30-60 λεπτά (ανάλογα με hardware)
- **Real-time Visualization**: Μπορεί να επιβραδύνει λίγο την εκπαίδευση
- **Metrics**: Αποθηκεύονται αυτόματα κάθε 5 επεισόδια
- **Checkpoints**: Αποθηκεύονται κάθε 50,000 timesteps

## 🎯 Αναμενόμενα Αποτελέσματα

Μετά την εκπαίδευση:
- **Success Rate**: >80% (ο πράκτορας φτάνει τον στόχο)
- **Mean Reward**: -100 έως -150 (αποδοτικά επεισόδια)
- **Mean Steps**: 100-150 βήματα
- **Learning Curve**: Βελτίωση από -200 (αποτυχία) σε -100 (επιτυχία)

---

**Τέλος Οδηγιών**


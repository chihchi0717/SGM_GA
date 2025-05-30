## SGM_GA
### 📘 README

Genetic Algorithm for Geometry Optimization in TracePro Simulations
This project implements a real-valued Genetic Algorithm (GA) to optimize geometric parameters for sunlight-guiding prism structures. The evaluation of each candidate geometry is conducted through automated CAD modeling in AutoCAD and ray-tracing simulation in TracePro.

### 📁 Project Structure
``` 
├── batched_main_elite_tournament_norepeat.py   # Main GA script with non-redundant modeling & simulation
├── draw_New.py                                 # Shape drawing logic
├── PYtoAutocad_New0523_light_center_short.py   # AutoCAD automation module
├── TracePro_fast.py                            # TracePro simulation module
├── txt_new.py                                  # Fitness evaluation logic
├── GA_population/                              # Folder to save CAD & simulation results
│   └── fitness_log.csv                         # Logged fitness results of all individuals per generation
``` 
### ⚙️ Features
Real-valued GA with:
    Uniform crossover
    Random mutation
    Tournament selection
    Elite retention (μ + λ strategy)

Automatic avoidance of repeated modeling/simulation via fitness_log.csv
Tracks and logs all individuals along with generation number for analysis
Compatible with AutoCAD + TracePro on 32-bit Python

### 🧪 Fitness Function
Fitness is defined as:
fitness = efficiency * (1 / (1 + process_score))

### 📝 CSV Format: fitness_log.csv
S1, S2, A1: Geometric parameters
fitness: Computed evaluation score
generation: The generation in which the individual was evaluated
💡 Repeated individuals across generations are allowed and logged separately.

### ▶️ How to Run
1. Ensure your environment includes:

    32-bit Python 3.13.3 (required by TracePro COM)
    AutoCAD and TracePro properly installed and licensed

2. Change the path
    

3. Run the scm script:
python scm.py

4. Run the GA script:
python batched_main_elite_tournament_norepeat.py

### 📌 Notes
Generation size and total generations can be configured via:

POP_SIZE = 100
N_GENERATIONS = 100

Output files and simulation folders are stored in GA_population/

Resume is supported by checking fitness_log.csv; duplicate modeling/simulation will be skipped.
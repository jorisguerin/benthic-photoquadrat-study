# Benthic Cover Estimation: Joint Effects of Sampling Design and Annotation Quality

This repository contains the complete code to reproduce the analysis from the research paper **"Assessing Joint Effects of Sampling Design and Annotation Quality on Benthic Cover Estimates through Monte Carlo Simulations"**.

## Installation

```bash
git clone git@github.com:XXX/benthic-photoquadrat-study.git
cd benthic-photoquadrat-study

python -m venv benthic-env
source benthic-env/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

Run scripts in this order:

```bash
# 1. Preprocess the map data
python scripts/preprocess_map.py

# 2. Run analysis scripts (any order)
python scripts/visualize_sampling_strategies.py
python scripts/quadrat_size_and_number_effects.py
python scripts/point_sampling_density_effects.py
python scripts/quadrat_placement_strategy_effects.py
python scripts/annotation_error_effects.py
python scripts/resource_allocation_strategy_comparison.py
```

Results saved in `figures/` directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{XXX,
  title={Assessing Joint Effects of Sampling Design and Annotation Quality on Benthic Cover Estimates through Monte Carlo Simulations},
  author={Anonymous},
  journal={[UNDER REVIEW]},
  year={2025}
}
```

## Contact

For questions about the code or research, please contact: anonymous

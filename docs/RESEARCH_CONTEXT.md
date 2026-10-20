# Research Context and Disclosure

This GitHub project is an **independent portfolio reimplementation**, not the original master's thesis software.

The academic research investigated machine-learning classification of iced vs non-iced outside heat exchanger conditions using real automotive MF4 measurements. Three prototype input signals were suction pressure, high refrigerant pressure and compressor speed.

In the thesis, XGBoost achieved 100% test accuracy on an 80/20 **sample-level** split. This is a reported result on the available measurement samples, not proof of generalization to new vehicles or climates. Additional recording evaluation indicated that restart transients could cause temporary false icing classifications.

The educational demonstration differs intentionally:
- Completely artificial vehicle-like measurements and class labels
- Independent synthetic parameter values and generic signal identifiers
- Fresh Python code (no original employer source files)
- Entire-recording holdout for validation rather than random row splitting
- No need for StandardScaler with tree-based XGBoost
- No actual thermal control integration, demonstrated defrost benefit, reduced PTC power use, or driving-range benefit

The thesis contributed to a **published Mercedes-Benz patent application**, not a granted patent. The portfolio code is not a production implementation of that invention.

Do not contribute OEM measurement logs, MF4 files, proprietary models, internal controller parameters, or confidential technical documentation to this repository.
